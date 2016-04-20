/**
 * 
 */
package fr.durandt.jstruct.ssvm.multiclass;

import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;


/**
 * 
 * ICML 2013
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class FastMulticlassSSVMFrankWolfe<X> extends FastMulticlassSSVM<X> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8900187396500347089L;

	protected int maxIter = 5;


	/**
	 * Train SSVM with Block-Coordinate Primal-Dual Frank-Wolfe
	 * @param l
	 */
	@Override
	protected void learning(List<STrainingSample<X, Integer>> l) {

		// init
		double lk = 0;
		double n = l.size();

		for(int k=0; k<maxIter; k++) {

			double[][] ws = new double[listClass.size()][dim];
			double ls = 0;

			for(int i=0; i<n; i++) {
				STrainingSample<X, Integer> ts = l.get(i);

				// loss augmented inference
				Integer yp = lossAugmentedInference(ts);

				VectorOp.add(ws[ts.output], psi(ts.input, ts.output));
				VectorOp.sub(ws[yp], psi(ts.input, yp));

				ls += delta(ts.output, yp);
			}

			VectorOp.mul(ws, 1/(lambda*n));
			ls /= n;

			double[][] diff = VectorOp.sub(w, ws);
			double gamma = (lambda * VectorOp.dot(diff, w) - lk + ls) / (lambda * VectorOp.dot(diff, diff));
			gamma = Math.min(Math.max(0, gamma),1);
			
			VectorOp.add(w, ws, 1-gamma, gamma);
			lk = (1-gamma)*lk + gamma*ls;

			if(verbose>=1) {
				System.out.println("epochs " + k + "/" + maxIter + "\tgamma= " + gamma);
			}
			else if(verbose>=2) {
				System.out.println("epochs " + k + "/" + maxIter + "\tgamma= " + gamma + "\tprimal obj= " + primalObj(l));
			}
			else {
				System.out.print(".");
			}
		}

		if(verbose == 0) {
			System.out.println("*");
		}
	}


	@Override
	protected void showParameters() {
		super.showParameters();
		System.out.println("Learning: Primal-Dual Frank-Wolfe - maxIter= " + maxIter);
	}

	@Override
	public String toString() {
		String s = "fast_multiclass_ssvm_FrankWolfe_lambda_" + lambda + "_maxIter_" + maxIter;
		return s;
	}

	/**
	 * @return the maxIter
	 */
	public int getMaxIter() {
		return maxIter;
	}

	/**
	 * @param maxIter the maxIter to set
	 */
	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}


}
