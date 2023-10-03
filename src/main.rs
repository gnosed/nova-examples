type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
use bellpepper::util_cs::witness_cs::SizedWitness;
use bellpepper_core::{
    boolean::AllocatedBit,
    num::{self, AllocatedNum},
    ConstraintSystem, SynthesisError, Variable,
};
use bincode::Error;
use ff::{derive::bitvec::vec, PrimeField};
use flate2::{write::ZlibEncoder, Compression};
use generic_array::typenum::U2;
use neptune::{
    circuit2::poseidon_hash_allocated,
    poseidon::{Poseidon, PoseidonConstants},
    Arity,
};
use nova_snark::{
    traits::{
        circuit::{StepCircuit, TrivialTestCircuit},
        Group,
    },
    CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigUint;
use pasta_curves::pallas::Scalar;
use std::{marker::PhantomData, time::Instant};

#[derive(Clone, Debug)]
struct HashStepCircuit<F: PrimeField> {
    _marker: PhantomData<F>,
}

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
        // use the inputs
        let v_0 = z[0].clone();
        let v_t = z[1].clone();
        let t = z[2].clone();

        let t_plus_1 = AllocatedNum::alloc(cs.namespace(|| format!("t_plus_1 = t + 1")), || {
            Ok(t.get_value().unwrap() + F::ONE)
        })?;

        let v_t_plus_1 = poseidon_hash_allocated(
            cs.namespace(|| "H(t||v_t)"),
            vec![t.clone(), v_t.clone()],
            &PoseidonConstants::<F, U2>::new(),
        )?;

        let b = AllocatedBit::alloc(
            cs.namespace(|| "b = t==0"),
            t.get_value().map(|t| t.is_zero_vartime()),
        )?;

        let q = AllocatedNum::alloc(cs.namespace(|| "q = (b+t)^-1"), || {
            let t_value = t.get_value().ok_or(SynthesisError::AssignmentMissing)?;
            let b_value = b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
            let b_field = if b_value { F::ONE } else { F::ZERO };

            Ok((t_value + b_field).invert().unwrap())
        })?;

        cs.enforce(
            || "b or t is zero",
            |lc| lc + b.get_variable(),
            |lc| lc + t.get_variable(),
            |lc| lc,
        );

        cs.enforce(
            || "b and t are not both zero",
            |lc| lc + b.get_variable() + t.get_variable(),
            |lc| lc + q.get_variable(),
            |lc| lc + CS::one(),
        );

        cs.enforce(
            || "if t == 0 or v_0 == v_t",
            |lc| lc + b.get_variable(),
            |lc| lc + v_0.get_variable() - v_t.get_variable(),
            |lc| lc,
        );

        cs.enforce(
            || "t_plus_1 == t + 1",
            |lc| lc + t.get_variable() + CS::one(),
            |lc| lc + CS::one(),
            |lc| lc + t_plus_1.get_variable(),
        );

        Ok(vec![v_0.clone(), v_t_plus_1.clone(), t_plus_1.clone()])
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
