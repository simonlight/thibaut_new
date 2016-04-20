/**
 * 
 */
package lsvmCCCPGazeVoc_PosNeg_Symil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.DoubleStream;

import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.latent.LatentRepresentationSymil;
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
	private static final long serialVersionUID = -3428682753907137329L;


	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
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
		
		setGroundTruthGazeMap(GroundTruthGazeRegion(l));
		
		for(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts : l) {
//			ts.sample.h = (int)(Math.random()*ts.sample.x.getInstances().size());
//			ts.sample.hp = 0;
//			ts.sample.hn = ts.sample.x.getInstances().size()-1;
			ts.sample.hp = (int)(Math.random()*ts.sample.x.getInstances().size());
			ts.sample.hn = (int)(Math.random()*ts.sample.x.getInstances().size());
			
			
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
		int hp = -1;
		double minVal = Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val < minVal) {
				minVal = val;
				hp = i;
			}
		}
		return hp;
	}
	
	public int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
	
	public Integer getGazeInitRegion(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>  ts, int scale, String mode){
		//Only positive image will be initialized by gaze most area
		//Negative image is initialized by 0
		if (mode.equals("+0")){
			if (ts.label == 1){
				Integer maxH = -1;
				double maxGazeRatio = -1;
				for (Integer h=0;h<convertScale(scale);h++){
					double gazeRatio = getPositiveGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio>=maxGazeRatio){
						maxH=h;
						maxGazeRatio = gazeRatio;
					}
				}
				return maxH;
			}
			
			else{
				return 0;
			}
		}
		else if(mode.equals("+-")){
			//Positive image is initialized by gaze most area
			//Negative image is initialized by gaze least area 
			if (ts.label==1){
				Integer maxH = -1;
				double maxGazeRatio = -1;
				for (Integer h=0;h<convertScale(scale);h++){
					double gazeRatio = getPositiveGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio>=maxGazeRatio){
						maxH=h;
						maxGazeRatio = gazeRatio;
					}
				}
				return maxH;
			}
			else{
				Integer minH = -1;
				double minGazeRatio = Integer.MAX_VALUE;
				for (Integer h=0;h<convertScale(scale);h++){
					double gazeRatio = getNegativeGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio<=minGazeRatio){
						minH=h;
						minGazeRatio = gazeRatio;
					}
				}
				return minH;
			}
		}
		
		else{
			//Default case: all initialized by 0
			return 0;
		} 
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
			return null;
		}
	}
	
	protected double getNegativeGazeRatio(BagImage x, Integer h, String gazeType){
		if (gazeType.equals("ferrari")){
			
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
			double gaze_ratio=0;
			for (String c: classes){
				if (lossMap.containsKey(c+"_"+ETLossFileName)){
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
			return null;
		}
	}
	
	public double getPositiveGazeLoss(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts, Integer h){
		
		return 1 - getPositiveGazeRatio(ts.sample.x, h, gazeType)
					/getPositiveGazeRatio(ts.sample.x,groundTruthGazeMap.get(ts.sample.x.getName()) , gazeType);
	}

	public double getNegativeGazeLoss(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts, Integer h){
		return 1 - getNegativeGazeRatio(ts.sample.x, h, gazeType)
					/getNegativeGazeRatio(ts.sample.x,groundTruthGazeMap.get(ts.sample.x.getName()) , gazeType);
	}

	@Override
	public HashMap<String , Integer> GroundTruthGazeRegion(List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> l) {
		

		HashMap<String , Integer> groundTruthGazeMap = new HashMap<String , Integer>(); 
		
		for(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts : l) {
			// ground truth gaze region of negative image is the one with the least gazes

			if (ts.label == 1){
				Integer maxH = -1;
				double maxGazeRatio = -1;
				for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
					double gazeRatio = getPositiveGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio>=maxGazeRatio){
						maxH=h;
						maxGazeRatio = gazeRatio;
					}
				}
				groundTruthGazeMap.put(ts.sample.x.getName(), maxH);
			}
			// ground truth gaze region of negative image is the one with the least gazes
			else if (ts.label == -1){
				Integer maxH = -1;
				double maxGazeRatio = -1;
				for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
					double negGazeRatio = getNegativeGazeRatio(ts.sample.x, h, gazeType);
					if (negGazeRatio>=maxGazeRatio){
						maxH=h;
						maxGazeRatio = negGazeRatio;
					}
				}
				groundTruthGazeMap.put(ts.sample.x.getName(), maxH);
				
			}
		}
		return groundTruthGazeMap;
	}
	
	public Object[] LAI(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts) {
		Integer hpredict = -1;
		double valmax = -Double.MAX_VALUE;
		Object[] lai = new Object[2];
		if (ts.label == 1){
			for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
				double val = getPositiveGazeLoss(ts, h) + valueOf(ts.sample.x, h);
				if(val>valmax){
					valmax = val;
					hpredict = h;
				}
			}
			lai[0] = hpredict;
			lai[1] = valmax;

		}
		else if (ts.label == -1){
			for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
				double val = getNegativeGazeLoss(ts, h) - valueOf(ts.sample.x, h);
				if(val>valmax){
					valmax = val;
					hpredict = h;
				}
			}
			lai[0] = hpredict;
			lai[1] = valmax;
		}

		return lai;
	}
	
	@Override
	public double[] getGazePsi(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts){
		double[] gazePsi= new double[dim];
		double[] laiPsi= psi(ts.sample.x, (Integer)LAI(ts)[0]);
		double[] gtGazePsi= psi(ts.sample.x, groundTruthGazeMap.get(ts.sample.x.getName()));
		for (int i =0; i<dim;i++){
			gazePsi[i] = laiPsi[i] - gtGazePsi[i];
		}
		return gazePsi;
	}

	@Override
	public double[] loss(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts) {
		
		Object[] lai = LAI(ts);
		Integer laiRegion = (Integer)lai[0];
		double laiValue = (double)lai[1];
		double g = laiValue - ts.label*valueOf(ts.sample.x, groundTruthGazeMap.get(ts.sample.x.getName()));
		double exNUM = nb[0]+nb[1];
		double positiveNUM = nb[0];
		double negativeNUM = nb[1];
		
		double[] lossTerm = new double[3];
//		if(ts.label == 1){
//			lossTerm[0] = Math.max(0, 1-valueOf(ts.sample.x, ts.sample.hp));
//			lossTerm[0] = lossTerm[0]; 
//			lossTerm[1] = g;
//			lossTerm[1] = lossTerm[1] * tradeoff;
//			lossTerm[2] = Math.max(0, 1 - (    valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)    )  ) ;
//			lossTerm[2] = lossTerm[2] * nbd;
//			return lossTerm;
//		}
//		else if (ts.label == -1){
//			lossTerm[0] = Math.max(0, 1+valueOf(ts.sample.x, ts.sample.hn));
//			lossTerm[0] = lossTerm[0] ; 
//			lossTerm[1] = g;
//			lossTerm[1] = lossTerm[1] * tradeoff;
//			lossTerm[2] = Math.max(0, 1 +(   valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)   )  ) ;
//			lossTerm[2] = lossTerm[2] *nbd;
//		
//			return lossTerm;
//		}

		if(ts.label == 1){
			lossTerm[0] = Math.max(0, 1-valueOf(ts.sample.x, ts.sample.hp));
			lossTerm[0] = lossTerm[0] / positiveNUM; 
			lossTerm[1] = g;
			lossTerm[1] = lossTerm[1] * tradeoff / positiveNUM;
			lossTerm[2] = Math.max(0, 1 - (    valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)    )  ) ;
			lossTerm[2] = lossTerm[2] * nbd  /  exNUM ;
			
			return lossTerm;
		}
		else if (ts.label == -1){
			lossTerm[0] = Math.max(0, 1+valueOf(ts.sample.x, ts.sample.hn));
			lossTerm[0] = lossTerm[0] / negativeNUM; 
			lossTerm[1] = g;
			lossTerm[1] = lossTerm[1] * tradeoff / negativeNUM;
			lossTerm[2] = Math.max(0, 1 +(   valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)   )  ) ;
			lossTerm[2] = lossTerm[2] *nbd /  exNUM ;
		
			return lossTerm;
		}
		return null;

	}
//	
	@Override
	public double getLoss(List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> l) {
		return null;
}
	
	@Override
	public double getLoss(List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> l, BufferedWriter trainingDetailFileOut) {
		double loss = 0;
		double positive_classfication_loss = 0;
		double negative_classfication_loss = 0;
		double positive_gaze_loss_bound = 0;
		double negative_gaze_loss_bound = 0;
		double positive_margin_loss = 0;
		double negative_margin_loss = 0;
		
		for(TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>> ts : l) {
			double[] example_loss = loss(ts);
			loss += DoubleStream.of(example_loss).sum();
			
			if (ts.label==1){
				positive_classfication_loss +=example_loss[0];
				positive_gaze_loss_bound +=example_loss[1];
				positive_margin_loss +=example_loss[2];
			}
			else{
				negative_classfication_loss +=example_loss[0];
				negative_gaze_loss_bound +=example_loss[1];
				negative_margin_loss +=example_loss[2];
			}
		}
		System.out.format("classification loss:%f, positive_gaze_loss_bound: %f, negative_gaze_loss_bound: %f,"
						+ " gaze_loss_bound:%f, positive_margin_loss:%f, negative_margin_loss:%f",
						positive_classfication_loss + negative_classfication_loss, positive_gaze_loss_bound, negative_gaze_loss_bound, 
						positive_gaze_loss_bound+negative_gaze_loss_bound, positive_margin_loss, negative_margin_loss);
		try {
			trainingDetailFileOut.write("classification_loss:"+(positive_classfication_loss + negative_classfication_loss)+
										" positive_gaze_loss_bound:"+positive_gaze_loss_bound+
										" negative_gaze_loss_bound:"+negative_gaze_loss_bound+
										" gaze_loss_bound:"+(positive_gaze_loss_bound+negative_gaze_loss_bound)+
										" positive_margin_loss:"+positive_margin_loss+
										" negative_margin_loss:"+negative_margin_loss);
			trainingDetailFileOut.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
//		loss /= l.size();
		return loss;
	}
	
	public double getScore(BagImage x){
		double maxScore = - Double.MAX_VALUE;
		double minScore = Double.MAX_VALUE;
		for(int h=0; h<x.getInstances().size(); h++) {
			double score_tmp = valueOf(x, h);
			if(score_tmp>maxScore){
				maxScore = score_tmp;
			}
			if(score_tmp<minScore){
				minScore = score_tmp;
			}
		}
		if (-minScore > maxScore){
			return minScore;
		}
		else{
			return maxScore;
		}
		
	}
	
	public double testAPRegion(List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> l, File resFile) {
		
		List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
			for(int i=0; i<l.size(); i++) {

				double pscore = valueOf(l.get(i).sample.x, l.get(i).sample.hp);
				double nscore = valueOf(l.get(i).sample.x, l.get(i).sample.hn);
				double score = pscore > -nscore ? pscore : nscore;
				Integer yp = score>0 ? 1 : -1;
				Integer hp = score>0 ? l.get(i).sample.hp : l.get(i).sample.hn; 
				Integer yi = l.get(i).label;
				out.write(Double.valueOf(score) + ","+Integer.valueOf(yp) +","+Integer.valueOf(yi) +","+ Integer.valueOf(hp)+","+l.get(i).sample.x.getName()+"\n");
				out.flush();
	        	eval.add(new Pair<Integer,Double>(yi , score)); 
	        }
			out.close();
	        	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}



}
