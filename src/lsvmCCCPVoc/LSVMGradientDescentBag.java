/**
 * 
 */
package lsvmCCCPVoc;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.util.Pair;
import fr.durandt.jstruct.util.AveragePrecision;
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
	private static final long serialVersionUID = -2073619978721969420L;

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<LatentRepresentation<BagImage, Integer>> copy()
			throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	@Override
	protected void init(List<TrainingSample<LatentRepresentation<BagImage, Integer>>> l) {
		dim = l.get(0).sample.x.getInstance(0).length;
		for(TrainingSample<LatentRepresentation<BagImage, Integer>> ts : l) {
//			ts.sample.h = (int)(Math.random()*ts.sample.x.getInstances().size());
			ts.sample.h = 0;
		}
	}

	@Override
	protected Integer optimizeH(BagImage x) {
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
	
	public int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
	
	
	@Override
	public double loss(TrainingSample<LatentRepresentation<BagImage, Integer>> ts) {
		double v = valueOf(ts.sample.x, ts.sample.h);
		if (ts.label == -1){
			return Math.max(0, 1 + v);
		}
		else if(ts.label == 1){
			return Math.max(1, v) - v;
		}
		return null;

	}
	
	public double testAP(List<TrainingSample<LatentRepresentation<BagImage,Integer>>> l) {
		
		List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
		for(int i=0; i<l.size(); i++) {
        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
        	double score = valueOf(l.get(i).sample.x,l.get(i).sample.h); // socre
//        	System.out.println(l.get(i).label+" "+score);
        	// label is changed to -1 1.
        	eval.add(new Pair<Integer,Double>((l.get(i).label), score)); //
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}	

}
