/**
 * 
 */
package lsvmAddGaze;

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
	private static final long serialVersionUID = 8563460344209868908L;

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
			ts.sample.h = 0;
		}
	}

	/*
	 * in the prediction phase, no fixations are used for scoring a region
	 * @see lsvm.LSVM#optimizePredictionH(java.lang.Object)
	 */
	@Override
	protected Integer optimizePredictionH(BagImage x) {
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
	protected Integer optimizePositiveH(BagImage x) {
		int hp = -1;
		double maxVal = -Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {

			double val = valueOf(x,i)+ tradeoff * getPositiveGazeRatio(x, i, gazeType);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
		}
		return hp;
	}

	@Override
	protected Integer optimizeNegativeH(BagImage x) {
		int hp = -1;
		double maxVal = -Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {
			
			double val = valueOf(x,i)+ tradeoff * getNegativeGazeRatio(x, i, gazeType);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
		}
		return hp;
	}

	
	protected double getPositiveGazeRatio(BagImage x, Integer h, String gazeType){
		if (gazeType.equals("ferrari")){
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = lossMap.get(className+"_"+ETLossFileName);

			return gaze_ratio;
		}
		else if (gazeType.equals("stefan")){
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = lossMap.get(ETLossFileName);

			return gaze_ratio;
		}
		else {
			System.err.println("error gazeType");
			return -1000.0;
		}
	}
	
	protected double getNegativeGazeRatio(BagImage x, Integer h, String gazeType){
		if (gazeType.equals("ferrari")){
			
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
			double gaze_ratio=0;
			int cnt=0;
			for (String c: classes){
				if (lossMap.containsKey(c+"_"+ETLossFileName)){
					cnt+=1;
					if (lossMap.get(c+"_"+ETLossFileName)>gaze_ratio){
						gaze_ratio =lossMap.get(c+"_"+ETLossFileName);
					} 
					
				}
			}
			return gaze_ratio;
		}
		else if (gazeType.equals("stefan")){
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = lossMap.get(ETLossFileName);

			return gaze_ratio;
		}
		else {
			System.err.println("error gazeType");
			return -1000.0;
		}
	}
	
	public double loss(TrainingSample<LatentRepresentation<BagImage, Integer>> ts) {
		double v = valueOf(ts.sample.x, ts.sample.h);
		if (ts.label == 1){
//			return Math.max(0, 1 - v);
			return Math.max(0, 1 - (v + tradeoff * getPositiveGazeRatio(ts.sample.x, ts.sample.h, gazeType)));
		}
		else{
			return Math.max(0, 1 + (v + tradeoff * getNegativeGazeRatio(ts.sample.x, ts.sample.h, gazeType)));
			}
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
