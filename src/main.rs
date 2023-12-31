type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

use bellpepper_core::{boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use generic_array::typenum::U2;
use neptune::{circuit2::poseidon_hash_allocated, poseidon::PoseidonConstants};
use nova_snark::{
    traits::{
        circuit::{StepCircuit, TrivialTestCircuit},
        Group,
    },
    PublicParams, RecursiveSNARK,
};
use std::{marker::PhantomData, time::Instant};

/// `HashStepCircuit` is a struct that represents a step in a hash function recursive circuit.
#[derive(Clone, Debug)]
struct HashStepCircuit<F: PrimeField> {
    _marker: PhantomData<F>,
}

/// Implementation of the `StepCircuit` trait for `HashStepCircuit`.
///
/// Given `v_1`, `v_t` and a number of steps `t` as the initial inputs
/// We define the step circuit v_{t+1} = H(t || v_t) where t starts at 1
/// The circuit receives the input vector z0 = [v_1, v_t, t] and outputs res = [v_1, v_{t+1}, t+1]
impl<F> StepCircuit<F> for HashStepCircuit<F>
where
    F: PrimeField,
{
    fn arity(&self) -> usize {
        3
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        // Use the inputs
        // The initial value `v_1` needs to be present at each step
        let v_1 = z[0].clone();
        let v_t = z[1].clone();
        let t = z[2].clone();

        let t_plus_1 = AllocatedNum::alloc(cs.namespace(|| format!("t_plus_1 = t + 1")), || {
            Ok(t.get_value().unwrap() + F::ONE)
        })?;
        // Calculate the hash H(t||v_t)
        let v_t_plus_1 = poseidon_hash_allocated(
            cs.namespace(|| "H(t||v_t)"),
            vec![t.clone(), v_t.clone()],
            &PoseidonConstants::<F, U2>::new(),
        )?;
        // Check the base case of the recursive circuit
        // Following the assignement instructions this should be t = 1 (F::ONE)
        let b = AllocatedBit::alloc(
            cs.namespace(|| "b = t==1"),
            t.get_value().map(|t| (t - F::ONE).is_zero_vartime()),
        )?;

        let q = AllocatedNum::alloc(cs.namespace(|| "q = (b+t)^-1"), || {
            let t_value = t.get_value().ok_or(SynthesisError::AssignmentMissing)?;
            let b_value = b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
            let b_field = if b_value { F::ONE } else { F::ZERO };

            Ok((t_value + b_field).invert().unwrap())
        })?;

        // Check the base case at t == 1
        cs.enforce(
            || "t and b are equal if t is 1 (base case)",
            |lc| lc + b.get_variable(),
            |lc| lc + t.get_variable() - b.get_variable(),
            |lc| lc,
        );
        // t cannot be equal to zero
        cs.enforce(
            || "b and t are not both zero",
            |lc| lc + b.get_variable() + t.get_variable(),
            |lc| lc + q.get_variable(),
            |lc| lc + CS::one(),
        );
        // If t == 1 (base case) then v_t should be the same as the initial value v_0
        cs.enforce(
            || "v_0 == v_t when t is base case",
            |lc| lc + b.get_variable(),
            |lc| lc + v_1.get_variable() - v_t.get_variable(),
            |lc| lc,
        );
        // Increase counter `t` by 1 at each step
        cs.enforce(
            || "t_plus_1 == t + 1",
            |lc| lc + t.get_variable() + CS::one(),
            |lc| lc + CS::one(),
            |lc| lc + t_plus_1.get_variable(),
        );

        Ok(vec![v_1.clone(), v_t_plus_1.clone(), t_plus_1.clone()])
    }
}

fn main() {
    println!("Nova-based Chain Hashing");
    println!("=========================================================");

    let num_steps = 4;
    let circuit_primary = HashStepCircuit {
        _marker: PhantomData,
    };

    let circuit_secondary = TrivialTestCircuit::default();
    println!("Proving {num_steps} of HashCircuit per step");

    // produce public parameters
    let start = Instant::now();
    println!("Producing public parameters...");
    let pp = PublicParams::<
        G1,
        G2,
        HashStepCircuit<<G1 as Group>::Scalar>,
        TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(&circuit_primary, &circuit_secondary);
    println!("PublicParams::setup, took {:?} ", start.elapsed());

    println!(
        "Number of constraints per step (primary circuit): {}",
        pp.num_constraints().0
    );
    println!(
        "Number of constraints per step (secondary circuit): {}",
        pp.num_constraints().1
    );

    println!(
        "Number of variables per step (primary circuit): {}",
        pp.num_variables().0
    );
    println!(
        "Number of variables per step (secondary circuit): {}",
        pp.num_variables().1
    );

    let z0_primary = vec![
        <G1 as Group>::Scalar::zero(),
        <G1 as Group>::Scalar::zero(),
        <G1 as Group>::Scalar::zero(),
    ];

    let z0_secondary = vec![<G2 as Group>::Scalar::zero()];

    type C1 = HashStepCircuit<<G1 as Group>::Scalar>;
    type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
    // produce a recursive SNARK
    println!("Generating a RecursiveSNARK...");
    let mut recursive_snark: RecursiveSNARK<G1, G2, C1, C2> = RecursiveSNARK::<G1, G2, C1, C2>::new(
        &pp,
        &circuit_primary,
        &circuit_secondary,
        z0_primary.clone(),
        z0_secondary.clone(),
    );

    for i in 0..num_steps {
        let start = Instant::now();
        let res = recursive_snark.prove_step(
            &pp,
            &circuit_primary,
            &circuit_secondary,
            z0_primary.clone(),
            z0_secondary.clone(),
        );
        assert!(res.is_ok());
        println!(
            "RecursiveSNARK::prove_step {}: {:?}, took {:?} ",
            i,
            res.is_ok(),
            start.elapsed()
        );
    }

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let start = Instant::now();
    let res = recursive_snark.verify(&pp, num_steps, &z0_primary, &z0_secondary);
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok());
}

#[cfg(test)]
mod tests {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    use crate::HashStepCircuit;
    use ff::{derive::rand_core::OsRng, Field};
    use nova_snark::{
        traits::{circuit::TrivialTestCircuit, Group},
        PublicParams, RecursiveSNARK,
    };
    use std::{marker::PhantomData, time::Instant};

    #[test]
    fn test_hash_step_circuit() {
        println!("Nova-based Chain Hashing");
        println!("=========================================================");

        let num_steps = 4;
        let circuit_primary = HashStepCircuit {
            _marker: PhantomData,
        };

        let circuit_secondary = TrivialTestCircuit::default();
        println!("Proving {num_steps} step of HashStepCircuit");

        let start = Instant::now();
        // produce public parameters
        println!("Producing public parameters...");
        let pp = PublicParams::<
            G1,
            G2,
            HashStepCircuit<<G1 as Group>::Scalar>,
            TrivialTestCircuit<<G2 as Group>::Scalar>,
        >::setup(&circuit_primary, &circuit_secondary);
        println!("PublicParams::setup, took {:?} ", start.elapsed());

        println!(
            "Number of constraints per step (primary circuit): {}",
            pp.num_constraints().0
        );
        println!(
            "Number of constraints per step (secondary circuit): {}",
            pp.num_constraints().1
        );

        println!(
            "Number of variables per step (primary circuit): {}",
            pp.num_variables().0
        );
        println!(
            "Number of variables per step (secondary circuit): {}",
            pp.num_variables().1
        );

        let v_0 = <G1 as Group>::Scalar::random(&mut OsRng);
        let t_0 = <G1 as Group>::Scalar::one();
        let z0_primary = vec![v_0, v_0, t_0];

        let z0_secondary = vec![<G2 as Group>::Scalar::zero()];

        type C1 = HashStepCircuit<<G1 as Group>::Scalar>;
        type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
        // produce a recursive SNARK
        println!("Generating a RecursiveSNARK...");
        let mut recursive_snark: RecursiveSNARK<G1, G2, C1, C2> =
            RecursiveSNARK::<G1, G2, C1, C2>::new(
                &pp,
                &circuit_primary,
                &circuit_secondary,
                z0_primary.clone(),
                z0_secondary.clone(),
            );

        for i in 0..num_steps {
            let start = Instant::now();
            let res = recursive_snark.prove_step(
                &pp,
                &circuit_primary,
                &circuit_secondary,
                z0_primary.clone(),
                z0_secondary.clone(),
            );
            assert!(res.is_ok());
            println!(
                "RecursiveSNARK::prove_step {}: {:?}, took {:?} ",
                i,
                res.is_ok(),
                start.elapsed()
            );
        }

        // verify the recursive SNARK
        println!("Verifying a RecursiveSNARK...");
        let start = Instant::now();
        let res = recursive_snark.verify(&pp, num_steps, &z0_primary, &z0_secondary);
        println!(
            "RecursiveSNARK::verify: {:?}, took {:?}",
            res.is_ok(),
            start.elapsed()
        );
        assert!(res.is_ok());
    }
}
