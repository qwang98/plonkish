use crate::{
    accumulation::{
        protostar::ProtostarStrategy::{Compressing, NoCompressing},
        PlonkishNark, PlonkishNarkInstance,
    },
    backend::PlonkishBackend,
    pcs::{AdditiveCommitment, PolynomialCommitmentScheme},
    poly::Polynomial,
    util::{
        arithmetic::{inner_product, powers, CurveAffine, Field},
        chain,
        expression::Expression,
        izip, izip_eq,
        transcript::Transcript,
        Deserialize, Itertools, Serialize,
    },
    Error,
};
use std::{iter, marker::PhantomData};

pub mod hyperplonk;
pub mod ivc;

// AccumulationScheme trait implemented for Protostar, with ProverParam, VerifierParam, Accumulator, AccumulatorInstance
#[derive(Clone, Debug)]
pub struct Protostar<Pb, const STRATEGY: usize = { Compressing as usize }>(PhantomData<Pb>);

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum ProtostarStrategy {
    // As known as Sangria
    NoCompressing = 0,
    // Compressing verification as described in 2023/620 section 3.5 but without square-root optimization
    #[default]
    Compressing = 1,
    // TODO:
    // Compressing verification with square-root optimization applied as described in 2023/620 section 3.5
    // CompressingWithSqrtPowers = 3,
}

impl From<usize> for ProtostarStrategy {
    fn from(strategy: usize) -> Self {
        [NoCompressing, Compressing][strategy]
    }
}
// accumulation prover???
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarProverParam<F, Pb>
where
    F: Field,
    Pb: PlonkishBackend<F>, // the backend that the prover is associated with. backend contains prover, verifier, and pcs
{
    pp: Pb::ProverParam, // is this the prover key from preprocessing???
    strategy: ProtostarStrategy, // accumulation verifier compression or not
    num_theta_primes: usize, // what are theta and alpha primes???
    num_alpha_primes: usize,
    num_folding_witness_polys: usize, // communication rounds k
    num_folding_challenges: usize, // communication rounds k-1 (no random challenge in the last round)
    cross_term_expressions: Vec<Expression<F>>, // error termss
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarVerifierParam<F, Pb>
where
    F: Field,
    Pb: PlonkishBackend<F>,
{
    vp: Pb::VerifierParam, // verifier key from preprocessing???
    strategy: ProtostarStrategy,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    num_cross_terms: usize,
}


// 3.4 of paper
#[derive(Clone, Debug)]
pub struct ProtostarAccumulator<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    instance: ProtostarAccumulatorInstance<F, Pcs::Commitment>, // accumulator instance acc.x
    witness_polys: Vec<Pcs::Polynomial>, // accumulator witness acc.w = m_i, where i = 1, ..., k, m_i are accumulated prover messages (polynomial)
    e_poly: Pcs::Polynomial, // error term polynomial, not mentioned in the paper, but its commitment is in ProtostarAccumulatorInstance
    _marker: PhantomData<Pcs>,
}

impl<F, Pcs> AsRef<ProtostarAccumulatorInstance<F, Pcs::Commitment>>
    for ProtostarAccumulator<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn as_ref(&self) -> &ProtostarAccumulatorInstance<F, Pcs::Commitment> {
        &self.instance
    }
}

impl<F, Pcs> ProtostarAccumulator<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn init(
        strategy: ProtostarStrategy,
        k: usize, // number of rounds is 2k-1
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        let zero_poly = Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]); // i think the zeros should be the coefficients, 2^k of them in total
        Self {
            instance: ProtostarAccumulatorInstance::init(
                strategy,
                num_instances,
                num_witness_polys, // this is the l number of polynomials in paper
                num_challenges, // this is the k number of challenges in paper
            ),
            witness_polys: iter::repeat_with(|| zero_poly.clone()) // creates infinite iterator of zero_poly and take the first num_witness_polys number of zero polies
                .take(num_witness_polys)
                .collect(),
            e_poly: zero_poly,
            _marker: PhantomData,
        }
    }

    fn from_nark(strategy: ProtostarStrategy, k: usize, nark: PlonkishNark<F, Pcs>) -> Self {
        let witness_polys = nark.witness_polys;
        Self {
            instance: ProtostarAccumulatorInstance::from_nark(strategy, nark.instance), // copies over the instance from nark, which has witness comms, challenges, u = 1, no error term commitment yet
            witness_polys, // copies over the witness polynomials from nark, m_i which is just polynomials
            e_poly: Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]), // error term polynomial, not in section 3.4 of paper, it's commitment is part of instance
            _marker: PhantomData,
        }
    }

    fn fold_uncompressed(
        &mut self,
        rhs: &Self,
        cross_term_polys: &[Pcs::Polynomial],
        cross_term_comms: &[Pcs::Commitment],
        r: &F,
    ) where
        Pcs::Commitment: AdditiveCommitment<F>,
    {
        self.instance
            .fold_uncompressed(&rhs.instance, cross_term_comms, r);
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), chain![cross_term_polys, [&rhs.e_poly]])
            .for_each(|(power_of_r, poly)| self.e_poly += (&power_of_r, poly));
    }

    fn fold_compressed(
        &mut self,
        rhs: &Self,
        zeta_cross_term_poly: &Pcs::Polynomial,
        zeta_cross_term_comm: &Pcs::Commitment,
        compressed_cross_term_sums: &[F],
        r: &F,
    ) where
        Pcs::Commitment: AdditiveCommitment<F>,
    {
        self.instance.fold_compressed(
            &rhs.instance,
            zeta_cross_term_comm,
            compressed_cross_term_sums,
            r,
        );
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), [zeta_cross_term_poly, &rhs.e_poly])
            .for_each(|(power_of_r, poly)| self.e_poly += (&power_of_r, poly));
    }

    pub fn instance(&self) -> &ProtostarAccumulatorInstance<F, Pcs::Commitment> {
        &self.instance
    }
}

// 3.4 of paper, accumulator instance acc.x
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarAccumulatorInstance<F, C> { // c is commitment
    instances: Vec<Vec<F>>,  // pi of length k in one dimension, accumulated public input
    witness_comms: Vec<C>,  // C_i of length k, accumulated commitments
    challenges: Vec<F>, // r_i of length k, accumulated challenges
    u: F, // slack variable
    e_comm: C, // accumulated commitment to error terms
    compressed_e_sum: Option<F>,
}

impl<F, C> ProtostarAccumulatorInstance<F, C> {
    fn instances(&self) -> &[Vec<F>] {
        &self.instances
    }
}

impl<F, C> ProtostarAccumulatorInstance<F, C>
where
    F: Field,
    C: Default,
{
    fn init(
        strategy: ProtostarStrategy,
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        Self {
            instances: num_instances.iter().map(|n| vec![F::ZERO; *n]).collect(),
            witness_comms: iter::repeat_with(C::default)
                .take(num_witness_polys)
                .collect(),
            challenges: vec![F::ZERO; num_challenges],
            u: F::ZERO,
            e_comm: C::default(),
            compressed_e_sum: match strategy {
                NoCompressing => None,
                Compressing => Some(F::ZERO),
            },
        }
    }

    fn claimed_sum(&self) -> F {
        self.compressed_e_sum.unwrap_or(F::ZERO)
    }

    fn absorb_into<CommitmentChunk>( // basically absorbs every element of the ProtostarAccumulatorInstance into PoseidonTranscript, so that we can squeeze out random challenges later
        &self,
        transcript: &mut impl Transcript<CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        C: AsRef<[CommitmentChunk]>,
    {
        self.instances // absorbs instances
            .iter()
            .try_for_each(|instances| transcript.common_field_elements(instances))?;
        self.witness_comms // these are committed coordinates of curve points after msm
            .iter()
            .try_for_each(|comm| transcript.common_commitments(comm.as_ref()))?; // PoseidonTranscript implementation of Transcript trait function, basically updates the Poseidon object state with coordinates of the committed point
        transcript.common_field_elements(&self.challenges)?; // absorbs challenges
        transcript.common_field_element(&self.u)?; // absorbs u
        transcript.common_commitments(self.e_comm.as_ref())?; // absorbs committed error term
        if let Some(compressed_e_sum) = self.compressed_e_sum.as_ref() {
            transcript.common_field_element(compressed_e_sum)?;
        }
        Ok(())
    }

    fn from_nark(strategy: ProtostarStrategy, nark: PlonkishNarkInstance<F, C>) -> Self {
        Self {
            instances: nark.instances, // pi from nark
            witness_comms: nark.witness_comms, // C_i from nark
            challenges: nark.challenges, // r_i from nark
            u: F::ONE, // u from nark, which is one
            e_comm: C::default(), // E from nark
            compressed_e_sum: match strategy {
                NoCompressing => None,
                Compressing => Some(F::ZERO),
            },
        }
    }

    fn fold_uncompressed(&mut self, rhs: &Self, cross_term_comms: &[C], r: &F)
    where
        C: AdditiveCommitment<F>,
    {
        let one = F::ONE;
        let powers_of_r = powers(*r).take(cross_term_comms.len() + 2).collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = C::sum_with_scalar([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        self.e_comm = {
            let comms = chain![[&self.e_comm], cross_term_comms, [&rhs.e_comm]];
            C::sum_with_scalar(&powers_of_r, comms)
        };
    }

    fn fold_compressed(
        &mut self,
        rhs: &Self,
        zeta_cross_term_comm: &C,
        compressed_cross_term_sums: &[F],
        r: &F,
    ) where
        C: AdditiveCommitment<F>,
    {
        let one = F::ONE;
        let powers_of_r = powers(*r)
            .take(compressed_cross_term_sums.len().max(1) + 2)
            .collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = C::sum_with_scalar([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        self.e_comm = {
            let comms = [&self.e_comm, zeta_cross_term_comm, &rhs.e_comm];
            C::sum_with_scalar(&powers_of_r[..3], comms)
        };
        *self.compressed_e_sum.as_mut().unwrap() += &inner_product(
            &powers_of_r[1..],
            chain![
                compressed_cross_term_sums,
                [rhs.compressed_e_sum.as_ref().unwrap()]
            ],
        );
    }
}

impl<F, Comm> ProtostarAccumulatorInstance<F, Comm>
where
    F: Field,
{
    pub fn unwrap_comm<C: CurveAffine>(self) -> ProtostarAccumulatorInstance<F, C>
    where
        Comm: AsRef<C>,
    {
        ProtostarAccumulatorInstance {
            instances: self.instances,
            witness_comms: chain![self.witness_comms.iter().map(AsRef::as_ref).copied()].collect(),
            challenges: self.challenges,
            u: self.u,
            e_comm: *self.e_comm.as_ref(),
            compressed_e_sum: self.compressed_e_sum,
        }
    }
}
