use crate::{
    accumulation::{
        protostar::{
            hyperplonk::{
                preprocessor::{batch_size, preprocess},
                prover::{
                    evaluate_compressed_cross_term_sums, evaluate_cross_term_polys,
                    evaluate_zeta_cross_term_poly, lookup_h_polys, powers_of_zeta_poly,
                },
            },
            ivc::ProtostarAccumulationVerifierParam,
            Protostar, ProtostarAccumulator, ProtostarAccumulatorInstance, ProtostarProverParam,
            ProtostarStrategy::{Compressing, NoCompressing},
            ProtostarVerifierParam,
        },
        AccumulationScheme, PlonkishNark, PlonkishNarkInstance,
    },
    backend::{
        hyperplonk::{
            prover::{
                instance_polys, lookup_compressed_polys, lookup_m_polys, permutation_z_polys,
                prove_sum_check,
            },
            verifier::verify_sum_check,
            HyperPlonk, HyperPlonkVerifierParam,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo,
    },
    pcs::{AdditiveCommitment, CommitmentChunk, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, PrimeField},
        end_timer, start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::BorrowMut, hash::Hash, iter};

mod preprocessor;
mod prover;

// protostar implements accumulationscheme (and is the only implementation here actually)
impl<F, Pcs, const STRATEGY: usize> AccumulationScheme<F> for Protostar<HyperPlonk<Pcs>, STRATEGY>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    Pcs::Commitment: AdditiveCommitment<F>,
    Pcs::CommitmentChunk: AdditiveCommitment<F>,
{
    type Pcs = Pcs;
    type ProverParam = ProtostarProverParam<F, HyperPlonk<Pcs>>;
    type VerifierParam = ProtostarVerifierParam<F, HyperPlonk<Pcs>>;
    type Accumulator = ProtostarAccumulator<F, Pcs>;
    type AccumulatorInstance = ProtostarAccumulatorInstance<F, Pcs::Commitment>;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<Pcs::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info, STRATEGY.into());
        Pcs::setup(poly_size, batch_size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(circuit_info.is_well_formed());

        preprocess(param, circuit_info, STRATEGY.into())
    }

    // create accumulator with accumulator instance (acc.x) and witness (acc.w), where witness is the witness polynomials, also has error term polynomial
    // instance contains commitments of witness and error term plus random challenges
    fn init_accumulator(pp: &Self::ProverParam) -> Result<Self::Accumulator, Error> {
        Ok(ProtostarAccumulator::init(
            pp.strategy,
            pp.pp.num_vars, // rounds
            &pp.pp.num_instances,
            pp.num_folding_witness_polys,
            pp.num_folding_challenges,
        ))
    }

    fn init_accumulator_from_nark(
        pp: &Self::ProverParam,
        nark: PlonkishNark<F, Self::Pcs>,
    ) -> Result<Self::Accumulator, Error> {
        Ok(ProtostarAccumulator::from_nark(
            pp.strategy,
            pp.pp.num_vars, // why 2^num_vars is the number of evals for the error polynomial???
            nark,
        ))
    }
    // implement AccumulationScheme trait function for Protostar
    fn prove_nark(
        pp: &Self::ProverParam, // is ProtostarProverParam in the test
        circuit: &impl PlonkishCircuit<F>, // PlonkishCircuit is a trait; in the test, the implementing struct is frontend::halo2::Halo2Circuit
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Pcs>, F>, // both P1 and P2 (accumulators) use strawman::PoseidonTranscript<_, _>, which implements TranscriptWrite
        _: impl RngCore,
    ) -> Result<PlonkishNark<F, Pcs>, Error> { // resulting PlonkishNark contains instance, challenges, witness commitment, and witness poly
        let ProtostarProverParam {
            pp,
            strategy,
            num_theta_primes,
            num_alpha_primes,
            ..
        } = pp;

        // the input `pp`
        // accumulation::protostar::ProtostarProverParam
        // pub struct ProtostarProverParam<F, Pb>
        // where
        //     F: Field,
        //     Pb: PlonkishBackend<F>, // the backend that the prover is associated with. backend contains prover, verifier, and pcs
        // {
        //     pp: Pb::ProverParam, // is this the prover key from preprocessing???
        //     strategy: ProtostarStrategy, // accumulation verifier compression or not
        //     num_theta_primes: usize, // what are theta and alpha primes???
        //     num_alpha_primes: usize,
        //     num_folding_witness_polys: usize, // communication rounds k
        //     num_folding_challenges: usize, // communication rounds k-1 (no random challenge in the last round)
        //     cross_term_expressions: Vec<Expression<F>>, // error termss
        // }

        // the input `circuit`
        // frontend::halo2::Halo2Circuit
        // pub struct Halo2Circuit<F: Field, C: Circuit<F>> {
        //     k: u32,
        //     instances: Vec<Vec<F>>,
        //     circuit: C,
        //     cs: ConstraintSystem<F>,
        //     config: C::Config,
        //     constants: Vec<Column<Fixed>>,
        //     num_witness_polys: Vec<usize>,
        //     advice_idx_in_phase: Vec<usize>,
        //     challenge_idx: Vec<usize>,
        //     row_mapping: Vec<usize>,
        // }

        // the input `transcript`
        // strawman::PoseidonTranscript
        // pub struct PoseidonTranscript<F: PrimeField, S> {
        //     state: Poseidon<F, T, RATE>,
        //     stream: S,
        // }

        // the transcript's poseidon object first absorbs instance variables, which is basically a preparation for r_0 (the first random challenge)
        // think of this as the prelude of r_0 <- \rho_NARK(pi) in section 3.3 of paper
        let instances = circuit.instances(); // instances are Vec<Vec<F>> for Halo2Circuit, each outter Vec represents a round
        for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) { // pp is the destructured HyperPlonkProverParam
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?; // basically loops over each F of instances: Vec<Vec<F>>; update the state of the poseidon object so that we can squeeze out challenges
            }
        }

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum()); // num_witness_polys and all other parameters of pp (HyperPlonkProverParam) are vectorized, seems there are multiple rounds of polynomials to communicate and each round communicates multiple polynomials with number specified by this vector
        let mut witness_comms = Vec::with_capacity(witness_polys.len()); // poly and comm have the same length
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum());
        // for each round, create multilinera polynomials out of inverted assigned witness values from synthesizing a circuit
        // commit these witness polynomials
        // extend the witness polynomials, their commitments, and challenges

        // multiple witness polynomials are collectively the witness of each round (w)
        
        // for m_i, which according to 3.3 is created by P_sps(pi, w, transcript), there's no message m_i created and committed (C_i) by prover
        // witness is directly commited as C_i, rather than comitting the prover's message in section 3.3, where C_i <- Commit(ck, m_i)
        // there's no commit key ck
        
        // randomness (challenge) generation: for each round, the transcript first absorbs the committed witness, and then simply absorbs its own hash output from last squeeze via `squeeze_challenges` before squeezing out another hash output
        // this is consistent with 3.3, which has r_i <- \rho_nark(r_{i-1}, C_i), basically the next randomness is created by random oracle (transcript) after feeding in the last randomness and commitment
        for (round, (num_witness_polys, num_challenges)) in pp // seems that each round is an element in the vectors, and each round can have multiple witness polys and challenges
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                // round number (index from enumerate) is fed into the `phase` parameter, `synthesize` returns a Vec<Vec<F>> of inverted witness assignment values (not sure why inverted)
                .synthesize(round, &challenges)? // challenges is an empty vector at the start; synthesize is PlonkishCircuit trait implementation, test example uses Halo2Circuit struct
                .into_iter()
                .map(MultilinearPolynomial::new) // the inverted witness assignment matrix is converted into multiple multilinear polynomials
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);
            // transcript absorbs committed witness
            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?); // batch commit and write commitments to transcript; commitment for P1 is univariateKZG, commited value is simply a value on a curve calculated from msm, the bases of msm are powers of a generator and the commited values are their scalar powers for the msm 
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges)); // call squeeze_challenge num_challenge times, implemented by PoseidonTranscript in test.rs; the squeeze_challenge function outputs hash result as a field element and updates the poseidon object's state input with the hash result, so that the next squeeze will output a different hash result; this is performed num_challenges times to generate num_challenge different "random" field elements as challenges
        }

        // Round n
        // lookup implementation, not reviewed yet
        let theta_primes = powers(transcript.squeeze_challenge()) // infinite iterator of powers of the squeezed challenge and take num_theta_primes many of the powers of challenge, the results are a vector of field elements
            .skip(1)
            .take(*num_theta_primes)
            .collect_vec();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let instance_polys = instance_polys(pp.num_vars, instances);
            let polys = iter::empty()
                .chain(instance_polys.iter())
                .chain(pp.preprocess_polys.iter())
                .chain(witness_polys.iter())
                .collect_vec();
            let thetas = iter::empty()
                .chain(Some(F::ONE))
                .chain(theta_primes.iter().cloned())
                .collect_vec();
            lookup_compressed_polys(&pp.lookups, &polys, &challenges, &thetas)
        };
        end_timer(timer);

        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?; // result of lookup are multilinear polynomials
        end_timer(timer);

        let lookup_m_comms = Pcs::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?; // write multilinear polynomial evals to transcript as committed curve point coordinates

        // Round n+1

        let beta_prime = transcript.squeeze_challenge(); // basically create a random number hashed from the transcript (fiat shamir)

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &beta_prime);
        end_timer(timer);

        let lookup_h_comms = {
            let polys = lookup_h_polys.iter().flatten();
            Pcs::batch_commit_and_write(&pp.pcs, polys, transcript)?
        };

        // Round n+2
        // only related to compressing strategy, not reviewed yet
        let (zeta, powers_of_zeta_poly, powers_of_zeta_comm) = match strategy {
            NoCompressing => (None, None, None),
            Compressing => {
                let zeta = transcript.squeeze_challenge();

                let timer = start_timer(|| "powers_of_zeta_poly");
                let powers_of_zeta_poly = powers_of_zeta_poly(pp.num_vars, zeta);
                end_timer(timer);

                let powers_of_zeta_comm =
                    Pcs::commit_and_write(&pp.pcs, &powers_of_zeta_poly, transcript)?;

                (
                    Some(zeta),
                    Some(powers_of_zeta_poly),
                    Some(powers_of_zeta_comm),
                )
            }
        };

        // Round n+3

        let alpha_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_alpha_primes)
            .collect_vec();
        // according to 3.3, the NARK sends over all commitments C_i and all prover messages m_i in all rounds
        // note that in this implementation, m_i is just the witness polynomials instead of m_i <- P_sps(pi, w, transcript) in 3.3
        // the witness commitments instead of message commitments are sent over as C_i
        // other than these, the NARK also sends over the public instances and challenges, which 3.3 didn't require, as challenges are theoretically all generated by verifier by reconstructing the transcript, and the instances should be available to both parties to start with
        Ok(PlonkishNark::new(
            instances.to_vec(), // copied from the original Halo2Circuit, whose instance is a Vec<Vec<>>, this keeps the same format, each outter Vec represents a different round
            // total length is: sum(num_challenges) + num_theta_prime + 1 + 1 + num_alpha_prime
            iter::empty()
                .chain(challenges) // length is sum of num_challenges, which is a Vec<usize>, where each usize is the num_challenges of a round
                .chain(theta_primes) // num_theta_prime many challenges, each an incremental power of a transcript squeezed randomness (the same randomness)
                .chain(Some(beta_prime)) // a single number
                .chain(zeta) // Single field element, None if no compressing and Some(transcript squeezed challenge) if some
                .chain(alpha_primes) // same as theta prime, but number is given by num_alpha_prime
                .collect(),
            iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms)
                .chain(powers_of_zeta_comm)
                .collect(),
            iter::empty()
                .chain(witness_polys)
                .chain(lookup_m_polys)
                .chain(lookup_h_polys.into_iter().flatten())
                .chain(powers_of_zeta_poly)
                .collect(),
        ))
    }
    // section 3.4 figure 3 accumulation prover
    // AccumulationScheme trait function implementation for Protostar
    // invoked by prove_accumulation_from_nark, which first invokes prove_nark and then invokes prove_accumulation according to AccumulationScheme trait
    fn prove_accumulation<const IS_INCOMING_ABSORBED: bool>(
        pp: &Self::ProverParam, // trait object, implemented as ProtostarProverParam here, which contains HyperPlonkProverParam as its pp field
        mut accumulator: impl BorrowMut<Self::Accumulator>, // ProtostarAccumulator, primary_acc, the running instance
        incoming: &Self::Accumulator, // incoming instance to fold, initiated from nark
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Pcs>, F>, // proof will be converted from transcript eventually
        _: impl RngCore,
    ) -> Result<(), Error> {
        let ProtostarProverParam {
            pp,
            strategy,
            num_alpha_primes,
            cross_term_expressions,
            ..
        } = pp;
        let accumulator = accumulator.borrow_mut();

        // ProtostarAccumulatorInstance implementation is in protostar.rs
        accumulator.instance.absorb_into(transcript)?; // ProtoStarAccumulatorInstance::absorb_into, basically absorbs all elements of ProtostarAccumulatorInstance (C_i, pi, r_i, u, E, etc.), so that we can squeeze out random challenge from the PoseidonTranscript later
        if !IS_INCOMING_ABSORBED {
            incoming.instance.absorb_into(transcript)?;
        }

        match strategy {
            NoCompressing => {
                let timer = start_timer(|| {
                    format!("evaluate_cross_term_polys-{}", cross_term_expressions.len())
                });
                let cross_term_polys = evaluate_cross_term_polys(
                    cross_term_expressions, // this is from ivc_pp.pp.cross_term_expressions, which is a Vec<Expressions>, obtained from preprocessing
                    pp.num_vars,
                    &pp.preprocess_polys, // MultilinearPolynomials
                    accumulator, // ProtostarAccumulator, primary_acc, the running instance, (acc)
                    incoming, // instance to fold, just converted from a nark to an accumulator at each folding step
                );
                end_timer(timer);

                let cross_term_comms =
                    Pcs::batch_commit_and_write(&pp.pcs, &cross_term_polys, transcript)?;

                // Round 0

                let r = transcript.squeeze_challenge();

                let timer = start_timer(|| "fold_uncompressed");
                accumulator.fold_uncompressed(incoming, &cross_term_polys, &cross_term_comms, &r);
                end_timer(timer);
            }
            Compressing => {
                let timer = start_timer(|| "evaluate_zeta_cross_term_poly");
                let zeta_cross_term_poly = evaluate_zeta_cross_term_poly(
                    pp.num_vars,
                    *num_alpha_primes,
                    accumulator,
                    incoming,
                );
                end_timer(timer);

                let timer = start_timer(|| {
                    let len = cross_term_expressions.len();
                    format!("evaluate_compressed_cross_term_sums-{len}")
                });
                let compressed_cross_term_sums = evaluate_compressed_cross_term_sums(
                    cross_term_expressions,
                    pp.num_vars,
                    &pp.preprocess_polys,
                    accumulator,
                    incoming,
                );
                end_timer(timer);

                let zeta_cross_term_comm =
                    Pcs::commit_and_write(&pp.pcs, &zeta_cross_term_poly, transcript)?;
                transcript.write_field_elements(&compressed_cross_term_sums)?;

                // Round 0

                let r = transcript.squeeze_challenge();

                let timer = start_timer(|| "fold_compressed");
                accumulator.fold_compressed(
                    incoming,
                    &zeta_cross_term_poly,
                    &zeta_cross_term_comm,
                    &compressed_cross_term_sums,
                    &r,
                );
                end_timer(timer);
            }
        }

        Ok(())
    }

    fn verify_accumulation_from_nark(
        vp: &Self::VerifierParam,
        mut accumulator: impl BorrowMut<Self::AccumulatorInstance>,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<CommitmentChunk<F, Self::Pcs>, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let ProtostarVerifierParam {
            vp,
            strategy,
            num_theta_primes,
            num_alpha_primes,
            num_cross_terms,
            ..
        } = vp;
        let accumulator = accumulator.borrow_mut();

        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum());
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n

        let theta_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_theta_primes)
            .collect_vec();

        let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1

        let beta_prime = transcript.squeeze_challenge();

        let lookup_h_comms = Pcs::read_commitments(&vp.pcs, 2 * vp.num_lookups, transcript)?;

        // Round n+2

        let (zeta, powers_of_zeta_comm) = match strategy {
            NoCompressing => (None, None),
            Compressing => {
                let zeta = transcript.squeeze_challenge();

                let powers_of_zeta_comm = Pcs::read_commitment(&vp.pcs, transcript)?;

                (Some(zeta), Some(powers_of_zeta_comm))
            }
        };

        // Round n+3

        let alpha_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_alpha_primes)
            .collect_vec();

        let nark = PlonkishNarkInstance::new(
            instances.to_vec(),
            iter::empty()
                .chain(challenges)
                .chain(theta_primes)
                .chain(Some(beta_prime))
                .chain(zeta)
                .chain(alpha_primes)
                .collect(),
            iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms)
                .chain(powers_of_zeta_comm)
                .collect(),
        );
        let incoming = ProtostarAccumulatorInstance::from_nark(*strategy, nark);
        accumulator.absorb_into(transcript)?;

        match strategy {
            NoCompressing => {
                let cross_term_comms =
                    Pcs::read_commitments(&vp.pcs, *num_cross_terms, transcript)?;

                // Round n+4

                let r = transcript.squeeze_challenge();

                accumulator.fold_uncompressed(&incoming, &cross_term_comms, &r);
            }
            Compressing => {
                let zeta_cross_term_comm = Pcs::read_commitment(&vp.pcs, transcript)?;
                let compressed_cross_term_sums =
                    transcript.read_field_elements(*num_cross_terms)?;

                // Round n+4

                let r = transcript.squeeze_challenge();

                accumulator.fold_compressed(
                    &incoming,
                    &zeta_cross_term_comm,
                    &compressed_cross_term_sums,
                    &r,
                );
            }
        };

        Ok(())
    }

    fn prove_decider(
        pp: &Self::ProverParam,
        accumulator: &Self::Accumulator,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Pcs>, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let ProtostarProverParam { pp, .. } = pp;

        accumulator.instance.absorb_into(transcript)?;

        // Round 0

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let builtin_witness_poly_offset = pp.num_witness_polys.iter().sum::<usize>();
        let instance_polys = instance_polys(pp.num_vars, &accumulator.instance.instances);
        let polys = iter::empty()
            .chain(&instance_polys)
            .chain(&pp.preprocess_polys)
            .chain(&accumulator.witness_polys[..builtin_witness_poly_offset])
            .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
            .collect_vec();
        let permutation_z_polys = permutation_z_polys(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let permutation_z_comms =
            Pcs::batch_commit_and_write(&pp.pcs, &permutation_z_polys, transcript)?;

        // Round 1

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = iter::empty()
            .chain(polys)
            .chain(&accumulator.witness_polys[builtin_witness_poly_offset..])
            .chain(permutation_z_polys.iter())
            .chain(Some(&accumulator.e_poly))
            .collect_vec();
        let challenges = iter::empty()
            .chain(accumulator.instance.challenges.iter().copied())
            .chain([accumulator.instance.u])
            .chain([beta, gamma, alpha])
            .collect();
        let (points, evals) = {
            prove_sum_check(
                pp.num_instances.len(),
                &pp.expression,
                accumulator.instance.claimed_sum(),
                &polys,
                challenges,
                y,
                transcript,
            )?
        };

        // PCS open

        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(pp.num_instances.len()))
            .chain(&pp.preprocess_comms)
            .chain(&accumulator.instance.witness_comms[..builtin_witness_poly_offset])
            .chain(&pp.permutation_comms)
            .chain(&accumulator.instance.witness_comms[builtin_witness_poly_offset..])
            .chain(&permutation_z_comms)
            .chain(Some(&accumulator.instance.e_comm))
            .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;
        end_timer(timer);

        Ok(())
    }

    fn verify_decider(
        vp: &Self::VerifierParam,
        accumulator: &Self::AccumulatorInstance,
        transcript: &mut impl TranscriptRead<CommitmentChunk<F, Pcs>, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let ProtostarVerifierParam { vp, .. } = vp;

        accumulator.absorb_into(transcript)?;

        // Round 0

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let permutation_z_comms =
            Pcs::read_commitments(&vp.pcs, vp.num_permutation_z_polys, transcript)?;

        // Round 1

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        let challenges = iter::empty()
            .chain(accumulator.challenges.iter().copied())
            .chain([accumulator.u])
            .chain([beta, gamma, alpha])
            .collect_vec();
        let (points, evals) = {
            verify_sum_check(
                vp.num_vars,
                &vp.expression,
                accumulator.claimed_sum(),
                accumulator.instances(),
                &challenges,
                &y,
                transcript,
            )?
        };

        // PCS verify

        let builtin_witness_poly_offset = vp.num_witness_polys.iter().sum::<usize>();
        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(vp.num_instances.len()))
            .chain(&vp.preprocess_comms)
            .chain(&accumulator.witness_comms[..builtin_witness_poly_offset])
            .chain(vp.permutation_comms.iter().map(|(_, comm)| comm))
            .chain(&accumulator.witness_comms[builtin_witness_poly_offset..])
            .chain(&permutation_z_comms)
            .chain(Some(&accumulator.e_comm))
            .collect_vec();
        Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;

        Ok(())
    }
}

impl<F, Pcs, N> From<&ProtostarVerifierParam<F, HyperPlonk<Pcs>>>
    for ProtostarAccumulationVerifierParam<N>
where
    F: PrimeField,
    N: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
    HyperPlonk<Pcs>: PlonkishBackend<F, VerifierParam = HyperPlonkVerifierParam<F, Pcs>>,
{
    fn from(vp: &ProtostarVerifierParam<F, HyperPlonk<Pcs>>) -> Self {
        let num_witness_polys = iter::empty()
            .chain(vp.vp.num_witness_polys.iter().cloned())
            .chain([vp.vp.num_lookups, 2 * vp.vp.num_lookups])
            .chain(match vp.strategy {
                NoCompressing => None,
                Compressing => Some(1),
            })
            .collect();
        let num_challenges = {
            let mut num_challenges = iter::empty()
                .chain(vp.vp.num_challenges.iter().cloned())
                .map(|num_challenge| vec![1; num_challenge])
                .collect_vec();
            num_challenges.last_mut().unwrap().push(vp.num_theta_primes);
            iter::empty()
                .chain(num_challenges)
                .chain([vec![1]])
                .chain(match vp.strategy {
                    NoCompressing => None,
                    Compressing => Some(vec![1]),
                })
                .chain([vec![vp.num_alpha_primes]])
                .collect()
        };
        Self {
            vp_digest: N::ZERO,
            strategy: vp.strategy,
            num_instances: vp.vp.num_instances.clone(),
            num_witness_polys,
            num_challenges,
            num_cross_terms: vp.num_cross_terms,
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        accumulation::{protostar::Protostar, test::run_accumulation_scheme},
        backend::hyperplonk::{
            util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_with_lookup_circuit},
            HyperPlonk,
        },
        pcs::{
            multilinear::{Gemini, MultilinearIpa, MultilinearKzg, Zeromorph},
            univariate::UnivariateKzg,
        },
        util::{
            test::{seeded_std_rng, std_rng},
            transcript::Keccak256Transcript,
            Itertools,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use std::iter;

    macro_rules! tests {
        ($name:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<$name _protostar_hyperplonk_vanilla_plonk>]() {
                    run_accumulation_scheme::<_, Protostar<HyperPlonk<$pcs>>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                        let circuits = iter::repeat_with(|| {
                            let (_, circuit) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                            circuit
                        }).take(3).collect_vec();
                        (circuit_info, circuits)
                    });
                }

                #[test]
                fn [<$name _protostar_hyperplonk_vanilla_plonk_with_lookup>]() {
                    run_accumulation_scheme::<_, Protostar<HyperPlonk<$pcs>>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _) = rand_vanilla_plonk_with_lookup_circuit(num_vars, std_rng(), seeded_std_rng());
                        let circuits = iter::repeat_with(|| {
                            let (_, circuit) = rand_vanilla_plonk_with_lookup_circuit(num_vars, std_rng(), seeded_std_rng());
                            circuit
                        }).take(3).collect_vec();
                        (circuit_info, circuits)
                    });
                }
            }
        };
        ($name:ident, $pcs:ty) => {
            tests!($name, $pcs, 2..16);
        };
    }

    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    tests!(kzg, MultilinearKzg<Bn256>);
    tests!(gemini_kzg, Gemini<UnivariateKzg<Bn256>>);
    tests!(zeromorph_kzg, Zeromorph<UnivariateKzg<Bn256>>);
}
