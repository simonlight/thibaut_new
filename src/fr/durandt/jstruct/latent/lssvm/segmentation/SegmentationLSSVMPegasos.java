/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.segmentation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.LSSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SegmentationLSSVMPegasos<X> extends LSSVMPegasos<X,Integer[],Integer[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2571716142412525593L;

	// List of categories
	protected List<Integer> listClass = null;

	protected int offsetY = -1;

	/**
	 * Return the representation of super-pixel
	 * @param x image
	 * @param sp index of super-pixel
	 * @return
	 */
	protected abstract double[] phi(X x, int sp);

	/**
	 * Return the representation of image x
	 * @param x
	 * @return
	 */
	protected abstract double[] phi(X x);

	/**
	 * Return the number of super-pixels
	 * @param x
	 * @return number of super-pixels
	 */
	protected abstract int getNumberOfSuperpixels(X x);

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(X x, Integer[] y, Integer[] h) {
		double[] psi = new double[dim];

		// h
		for(int i=0; i<h.length; i++) {
			double[] phi = phi(x,i);
			int offset = h[i] * phi.length;
			for(int d=0; d<phi.length; d++) {
				psi[d + offset] += phi[d];
			}
		}

		// y
		double[] phi = phi(x);
		for(int i=0; i<y.length; i++) {
			int offset = i * phi.length + offsetY;
			if(y[i] == 1) {
				for(int d=0; d<phi.length; d++) {
					psi[d + offset] += phi[d];
				}
			}
			else {
				for(int d=0; d<phi.length; d++) {
					psi[d + offset] -= phi[d];
				}
			}
		}

		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(Integer[] y, Integer[] yp, Integer[] hp) {
		double loss = 0;
		for(int i=0; i<y.length; i++) {
			loss += delta(y[i], yp[i]);
		}
		return loss;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X, Integer[]>, Integer[]> ts, double[] w) {

		// number of labels/states
		int numStates = listClass.size();
		// number of super-pixels
		int numNodes = getNumberOfSuperpixels(ts.input.x);

		// Create factor graph.
		FactorGraph factorGraph = new FactorGraph();

		// variables h
		int numVariables = numStates * numNodes;
		List<BinaryVariable> variables = new ArrayList<BinaryVariable>(numVariables);
		for(int i=0; i<numNodes; i++) {
			double[] phi = phi(ts.input.x, i);
			for(int j=0; j<numStates; j++) {
				double logPotential = valueOf(w, phi, j*phi.length);
				BinaryVariable variable = factorGraph.createBinaryVariable();
				variable.setLogPotential(logPotential);
				variables.add(variable);
			}
		}

		// variables y
		double[] phi = phi(ts.input.x);
		for(int j=0; j<numStates; ++j) {
			double logPotential = valueOf(w, phi, j*phi.length + offsetY) + delta(ts.output[j], 1); 
			BinaryVariable variable = factorGraph.createBinaryVariable();
			variable.setLogPotential(logPotential);
			variables.add(variable);
		}

		// XOR factor for multi-label variables
		for(int i=0; i<numNodes; i++) {
			List<BinaryVariable> binaryVariables = new ArrayList<BinaryVariable>(numStates);
			List<Boolean> negated = new ArrayList<Boolean>(numStates);
			for(int j=0; j<numStates; j++) {
				negated.add(false);
				binaryVariables.add(j, variables.get(j + i*numStates));
			}
			factorGraph.createFactorXOR(binaryVariables, negated);
		}

		// OROUT factor to link y and h
		for(int i=0; i<numStates; i++) {
			List<BinaryVariable> binaryVariables = new ArrayList<BinaryVariable>(numStates);
			List<Boolean> negated = new ArrayList<Boolean>(numStates);
			for(int j=0; j<=numNodes; j++) {
				negated.add(false);
				binaryVariables.add(j, variables.get(i + j*numStates));
			}
			factorGraph.createFactorOROUT(binaryVariables, negated);
		}

		//factorGraph.print();

		AD3 solver = new AD3();
		solver.setVerbose(0);
		solver.setEta(0.1);
		solver.setMaxIterations(1000);
		solver.setAdaptEta(true);

		List<Double> posteriors = new ArrayList<Double>();
		List<Double> additionalPosteriors = new ArrayList<Double>();
		double[] value = {0.0};
		//solver.solveLPMAP(factorGraph, posteriors, additionalPosteriors, value);
		solver.solveExactMAP(factorGraph, posteriors, additionalPosteriors, value);

		Integer[] ypredict = getBestConfigurationOutput(numNodes, numStates, variables, posteriors);
		Integer[] hpredict = getBestConfigurationLatent(numNodes, numStates, variables, posteriors);

		if(verbose>1) {
			System.out.println("lai - ypredict= " + Arrays.toString(ypredict));
			System.out.println("lai - hpredict= " + Arrays.toString(hpredict));
			System.out.println("lai - posteriors= " + posteriors);
		}

		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected Integer[] prediction(X x, Integer[] y, double[] w) {

		// number of labels/states
		int numStates = listClass.size();
		// number of super-pixels
		int numNodes = getNumberOfSuperpixels(x);

		// Create factor graph.
		FactorGraph factorGraph = new FactorGraph();

		// variables h
		int numVariables = numStates * numNodes;
		List<BinaryVariable> variables = new ArrayList<BinaryVariable>(numVariables);
		for(int i=0; i<numNodes; i++) {
			double[] phi = phi(x, i);
			for(int j=0; j<numStates; j++) {
				double logPotential = valueOf(w, phi, j*phi.length);
				BinaryVariable variable = factorGraph.createBinaryVariable();
				variable.setLogPotential(logPotential);
				variables.add(variable);
			}
		}

		// variables y
		for(int j=0; j<numStates; ++j) {
			double logPotential = (y[j] == 1 ? 1e10 : 0.); 
			BinaryVariable variable = factorGraph.createBinaryVariable();
			variable.setLogPotential(logPotential);
			variables.add(variable);
		}

		// XOR factor for multi-label variables
		for(int i=0; i<numNodes; i++) {
			List<BinaryVariable> binaryVariables = new ArrayList<BinaryVariable>(numStates);
			List<Boolean> negated = new ArrayList<Boolean>(numStates);
			for(int j=0; j<numStates; j++) {
				negated.add(false);
				binaryVariables.add(j, variables.get(j + i*numStates));
			}
			factorGraph.createFactorXOR(binaryVariables, negated);
		}

		// OROUT factor to link y and h
		for(int i=0; i<numStates; i++) {
			List<BinaryVariable> binaryVariables = new ArrayList<BinaryVariable>(numStates);
			List<Boolean> negated = new ArrayList<Boolean>(numStates);
			for(int j=0; j<=numNodes; j++) {
				negated.add(false);
				binaryVariables.add(j, variables.get(i + j*numStates));
			}
			factorGraph.createFactorOROUT(binaryVariables, negated);
		}

		//factorGraph.print();

		AD3 solver = new AD3();
		solver.setVerbose(0);
		solver.setEta(0.1);
		solver.setMaxIterations(1000);
		solver.setAdaptEta(true);

		List<Double> posteriors = new ArrayList<Double>();
		List<Double> additionalPosteriors = new ArrayList<Double>();
		double[] value = {0.0};
		//solver.solveLPMAP(factorGraph, posteriors, additionalPosteriors, value);
		solver.solveExactMAP(factorGraph, posteriors, additionalPosteriors, value);

		Integer[] ypredict = getBestConfigurationOutput(numNodes, numStates, variables, posteriors);
		Integer[] hpredict = getBestConfigurationLatent(numNodes, numStates, variables, posteriors);

		if(verbose>1) {
			System.out.println("prediction - y= " + Arrays.toString(y));
			System.out.println("prediction - ypredict= " + Arrays.toString(ypredict));
			System.out.println("prediction - hpredict= " + Arrays.toString(hpredict));
			System.out.println("prediction - posteriors= " + posteriors);
		}

		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#predictionOutputLatent(java.lang.Object, double[])
	 */
	@Override
	protected Object[] predictionOutputLatent(X x, double[] w) {

		// number of labels/states
		int numStates = listClass.size();
		// number of super-pixels
		int numNodes = getNumberOfSuperpixels(x);

		// Create factor graph.
		FactorGraph factorGraph = new FactorGraph();

		// variables h
		int numVariables = numStates * numNodes;
		List<BinaryVariable> variables = new ArrayList<BinaryVariable>(numVariables);
		for(int i=0; i<numNodes; i++) {
			double[] phi = phi(x, i);
			for(int j=0; j<numStates; j++) {
				double logPotential = valueOf(w, phi, j*phi.length);
				BinaryVariable variable = factorGraph.createBinaryVariable();
				variable.setLogPotential(logPotential);
				variables.add(variable);
			}
		}

		// variables y
		double[] phi = phi(x);
		for(int j=0; j<numStates; ++j) {
			double logPotential = valueOf(w, phi, j*phi.length + offsetY); 
			BinaryVariable variable = factorGraph.createBinaryVariable();
			variable.setLogPotential(logPotential);
			variables.add(variable);
		}

		// XOR factor for multi-label variables
		for(int i=0; i<numNodes; i++) {
			List<BinaryVariable> binaryVariables = new ArrayList<BinaryVariable>(numStates);
			List<Boolean> negated = new ArrayList<Boolean>(numStates);
			for(int j=0; j<numStates; j++) {
				negated.add(false);
				binaryVariables.add(j, variables.get(j + i*numStates));
			}
			factorGraph.createFactorXOR(binaryVariables, negated);
		}

		// OROUT factor to link y and h
		for(int i=0; i<numStates; i++) {
			List<BinaryVariable> binaryVariables = new ArrayList<BinaryVariable>(numStates);
			List<Boolean> negated = new ArrayList<Boolean>(numStates);
			for(int j=0; j<=numNodes; j++) {
				negated.add(false);
				binaryVariables.add(j, variables.get(i + j*numStates));
			}
			factorGraph.createFactorOROUT(binaryVariables, negated);
		}

		//factorGraph.print();

		AD3 solver = new AD3();
		solver.setVerbose(0);
		solver.setEta(0.1);
		solver.setMaxIterations(1000);
		solver.setAdaptEta(true);

		List<Double> posteriors = new ArrayList<Double>();
		List<Double> additionalPosteriors = new ArrayList<Double>();
		double[] value = {0.0};
		//solver.solveLPMAP(factorGraph, posteriors, additionalPosteriors, value);
		solver.solveExactMAP(factorGraph, posteriors, additionalPosteriors, value);

		Integer[] ypredict = getBestConfigurationOutput(numNodes, numStates, variables, posteriors);
		Integer[] hpredict = getBestConfigurationLatent(numNodes, numStates, variables, posteriors);

		if(verbose>1) {
			System.out.println("prediction - ypredict= " + Arrays.toString(ypredict));
			System.out.println("prediction - hpredict= " + Arrays.toString(hpredict));
			System.out.println("prediction - posteriors= " + posteriors);
		}

		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected Integer[] prediction(LatentRepresentation<X, Integer[]> x, double[] w) {
		Object[] res = predictionOutputLatent(x.x, w);
		return (Integer[]) res[0];
	}

	/**
	 * Compute the loss for 1 category
	 * @param y
	 * @param yp
	 * @return
	 */
	protected double delta(Integer y, Integer yp) {
		if(y != yp) {
			return 100.0;
		}
		else {
			return 0.0;
		}
	}

	protected double valueOf(double[] w, double[] psi, int offset) {
		double val = 0;
		for(int i=0; i<psi.length; i++) {
			val += w[i+offset] * psi[i];
		}
		return val;
	}

	protected Integer[] getBestConfigurationOutput(int numNodes, int numStates, List<BinaryVariable> variables, List<Double> posteriors) {
		int offset = numNodes*numStates;
		Integer[] bestStates = new Integer[numStates];
		for(int i=0; i<numStates; ++i) {
			bestStates[i] = (posteriors.get(i+offset)>0.5 ? 1 : 0);
		}
		return bestStates;
	}

	protected Integer[] getBestConfigurationLatent(int numNodes, int numStates, List<BinaryVariable> variables, List<Double> posteriors) {
		int offset=0;
		Integer[] bestStates = new Integer[numNodes];
		for(int i=0; i<numNodes; ++i) {
			int best = -1;
			for(int k = 0; k < numStates; ++k) {
				if (best < 0 || posteriors.get(offset+k) > posteriors.get(offset+best)) {
					best = k;
				}
			}
			offset += numStates;
			bestStates[i] = best;
		}
		return bestStates;
	}
}
