/**
 * 
 */
package Symil_Musk;

import java.util.List;
import java.util.stream.DoubleStream;

import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.latent.LatentRepresentationSymil;
import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVMGradientDescentBag extends LSVMGradientDescent<BagImage,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8563460344209868908L;

	@Override
	public Classifier<LatentRepresentationSymil<BagImage, Integer,Integer>> copy()
			throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	@Override
	protected void init(List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> l) {
		dim = l.get(0).sample.x.getInstance(0).length;
		
		for(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts : l) {
//			ts.sample.h = (int)(Math.random()*ts.sample.x.getInstances().size());
//			ts.sample.hp = (int)(Math.random()*ts.sample.x.getInstances().size());
//			ts.sample.hn = (int)(Math.random()*ts.sample.x.getInstances().size());
			
			ts.sample.hp = 0;
			ts.sample.hn = ts.sample.x.getInstances().size() - 1;
//			ts.sample.h = groundTruthGazeMap.get(ts.sample.x.getName());
		}
	}

	@Override
	protected Integer[]  optimizeH(BagImage x) {
		int hp = -1;
		int hn = -1;
		double maxVal = -Double.MAX_VALUE;
		double minVal = Double.MAX_VALUE;
		
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
			if(val < minVal) {
				minVal = val;
				hn = i;
			}
		}
		Integer[] hpredict = {hp,hn};
		return hpredict;
	}
	@Override
	protected Integer optimizePositiveH(BagImage x) {
		int hp = -1;
		double maxVal = -Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
		}
		return hp;
	}
	@Override
	protected Integer optimizeNegativeH(BagImage x) {
		int hn = -1;
		double minVal = Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val < minVal) {
				minVal = val;
				hn = i;
			}
		}
		return hn;
	}
	
@Override
public double[] loss(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts) {
		
		double exNUM = nb[0]+nb[1];
		double positiveNUM = nb[0];
		double negativeNUM = nb[1];
		
		double[] lossTerm = new double[2];
		if(ts.label == 1){
			lossTerm[0] = Math.max(0, 1-valueOf(ts.sample.x, ts.sample.hp));
			lossTerm[0] = lossTerm[0] / positiveNUM; 
			lossTerm[1] = Math.max(0, 1 - (    valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)    )  ) ;
			lossTerm[1] = lossTerm[1] *nbd /  exNUM;
		}
		else if (ts.label == -1){
			lossTerm[0] = Math.max(0, 1+valueOf(ts.sample.x, ts.sample.hn));
			lossTerm[0] = lossTerm[0] / negativeNUM; 
			lossTerm[1] = Math.max(0, 1 +(   valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)   )  ) ;
			lossTerm[1] = lossTerm[1] *nbd /  exNUM;
		}
		return lossTerm;


	}

@Override
public double getLoss(List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> l) {
		double loss = 0;
		for(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts : l) {
			double[] example_loss = loss(ts);
			loss += DoubleStream.of(example_loss).sum();
		}
		return loss/l.size();

}

	
}
