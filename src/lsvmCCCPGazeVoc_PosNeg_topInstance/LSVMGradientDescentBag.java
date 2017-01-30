/**
 * 
 */
package lsvmCCCPGazeVoc_PosNeg_topInstance;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.DoubleStream;

import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.latent.LatentRepresentationTopK;
import fr.durandt.jstruct.util.Pair;
import fr.durandt.jstruct.util.AveragePrecision;
import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSVMGradientDescentBag extends LSVMGradientDescent<BagImage,Integer> {

	private static final long serialVersionUID = -3428682753907137329L;


	@Override
	public Classifier<LatentRepresentationTopK<BagImage, Integer>> copy()
			throws CloneNotSupportedException {
		return null;
	}
	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	@Override
	protected void init(List<TrainingSample<LatentRepresentationTopK<BagImage, Integer>>> l) {
		dim = l.get(0).sample.x.getInstance(0).length;
		
		setGroundTruthGazeMap(GroundTruthGazeRegion(l));
		
		for(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts : l) {
			//add k instances 
			for(int cnt = 0; cnt < this.K; cnt += 1) {
				ts.sample.hlist.add(groundTruthGazeMap.get(ts.sample.x.getName()).get(cnt));
			}
		}
	}

	@Override
	protected ArrayList<Integer> optimizeH(BagImage x) {
		ArrayList<Double> epochHValList = new ArrayList<Double>();
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			epochHValList.add(val);
		}
		//save epochHValList
		ArrayList<Double> epochHValStoreList = new ArrayList<Double>(epochHValList); // may need to be new ArrayList(nfit)
		//sort epochHValList  by descent order
		Collections.sort(epochHValList ,Collections.reverseOrder());
		
		ArrayList<Integer> Hindexes = new ArrayList<Integer>();

		//K max instance indexs
		for (int n = 0; n < this.K; n++){
			Hindexes.add(epochHValStoreList.indexOf(epochHValList.get(n)));
			//wipe the value already indexed out.
			epochHValStoreList.set(epochHValStoreList.indexOf(epochHValList.get(n)), Double.NaN);
		}

		return Hindexes;
	}
	
	public int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
	
	protected double getGazeRatio(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts, Integer h, String gazeType){
		if (gazeType.equals("ferrari") ){
			if(ts.label==1){
				String featurePath[] = ts.sample.x.getInstanceFile(h).split("/");
				String ETLossFileName = featurePath[featurePath.length - 1];
				if (gazeRatioMap.containsKey(className+"_"+ETLossFileName)){
					double gaze_ratio = gazeRatioMap.get(className+"_"+ETLossFileName);
					return gaze_ratio;
				}
				else{
					return 1.;
				}
			}
			else{
				/*	only the region with high gaze ratio has a high probability of containing an object
					here we compare the gaze ratio across all possible categories, and takes the maximum
					gaze ratio as the gaze ratio
				 * 
				 */
				String featurePath[] = ts.sample.x.getInstanceFile(h).split("/");
				//filename or fileroot, it does not have the category name.
				String ETLossFileName = featurePath[featurePath.length - 1];
				String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
				double gaze_ratio=-1.0;
				for (String c: classes){
					if (gazeRatioMap.containsKey(c+"_"+ETLossFileName)){
						if (gazeRatioMap.get(c+"_"+ETLossFileName)>gaze_ratio){
							gaze_ratio =gazeRatioMap.get(c+"_"+ETLossFileName);
						} 
					}
				}
				return gaze_ratio;
			}
		}
		else if (gazeType.equals("stefan")){
			String featurePath[] = ts.sample.x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = gazeRatioMap.get(ETLossFileName);
			return gaze_ratio;
		}
		else if (gazeType.equals("ufood")){
			String featurePath[] = ts.sample.x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = gazeRatioMap.get(ETLossFileName);

			return gaze_ratio;
		}
		else {
			System.err.println("error gazeType");
			return (Double)null;
		}
	}
	
	public double getPositiveGazeLoss(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts, Integer h){
		double thisRatio = getGazeRatio(ts, h, gazeType);
		double gtRatio = getGazeRatio(ts,groundTruthGazeMap.get(ts.sample.x.getName()).get(0), gazeType);
		return 1 - (thisRatio / gtRatio);
	}

	public double getNegativeGazeLoss(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts, Integer h){
		double thisRatio = getGazeRatio(ts, h, gazeType);
		double gtRatio = getGazeRatio(ts,groundTruthGazeMap.get(ts.sample.x.getName()).get(convertScale(this.scale)-1) , gazeType);
//		double gtRatio = getGazeRatio(ts,groundTruthGazeMap.get(ts.sample.x.getName()).get(0) , gazeType);
		double minRatio = getGazeRatio(ts,groundTruthGazeMap.get(ts.sample.x.getName()).get(0) , gazeType);
		return (thisRatio - minRatio) / (gtRatio - minRatio+0.0000001);
//		return 1 - (thisRatio / gtRatio);
		}

	@Override
	public HashMap<String, ArrayList<Integer>> GroundTruthGazeRegion(List<TrainingSample<LatentRepresentationTopK<BagImage, Integer>>> l) {
		HashMap<String , ArrayList<Integer>> groundTruthGazeMap = new HashMap<String , ArrayList<Integer>>(); 
		for(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts : l) {
			ArrayList<Double> glList = new ArrayList<Double>();
				for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
					double gazeRatio = getGazeRatio(ts, h, gazeType);
					glList.add(gazeRatio);
				}
				ArrayList<Double> glStoreList = new ArrayList<Double>(glList);
				//the order of the ground truth regions
				//if + : descent
				//if - : ascent
				Collections.sort(glList, ts.label==1?Collections.reverseOrder():null);
				//the order of the ground truth regions
				//if + : descent
				//if - : descent
//				Collections.sort(glList, Collections.reverseOrder());

				ArrayList<Integer> Hindexes = new ArrayList<Integer>();
				//keep a list for all groundtruth regions, because negative image needs both ends
				for (int n = 0; n < convertScale(this.scale); n++){
					Hindexes.add(glStoreList.indexOf(glList.get(n)));
					//wipe the value already indexed out.
					glStoreList.set(glStoreList.indexOf(glList.get(n)), Double.NaN);
				}
				groundTruthGazeMap.put(ts.sample.x.getName(), Hindexes);
		}
		return groundTruthGazeMap;
	}
	
	public Object[] LAI(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts) {
		double valMax = 0;
		Object[] lai = new Object[2];
		ArrayList<Double> laiList = new ArrayList<Double>();
		for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
			double laiValue = (ts.label==1?getPositiveGazeLoss(ts, h):getNegativeGazeLoss(ts, h))+valueOf(ts.sample.x, h);
			laiList.add(laiValue);
		}
		ArrayList<Double> laiStoreList = new ArrayList<Double>(laiList);
		//LAI is always descent order
		Collections.sort(laiList, Collections.reverseOrder());
		ArrayList<Integer> Hindexes = new ArrayList<Integer>();
		for (int n = 0; n < this.K; n++){
			Hindexes.add(laiStoreList.indexOf(laiList.get(n)));
			valMax+=(laiList.get(n)/this.K);
			//wipe the value already indexed out.
			laiStoreList.set(laiStoreList.indexOf(laiList.get(n)), Double.NaN);
		}
		lai[0] = Hindexes;
		lai[1] = valMax;
		return lai;
	}
	
	@Override
	public double[] getGazePsi(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts){
		//generate the gradient
		double[] gazePsi= new double[dim];
		ArrayList<Integer> laiRegions = (ArrayList<Integer>)LAI(ts)[0];
		ArrayList<Integer> gtGazeRegions = groundTruthGazeMap.get(ts.sample.x.getName());
		for (int cnt=0; cnt<this.K;cnt++){
			double[] laiFeature = psi(ts.sample.x, laiRegions.get(cnt));
			double[] gtGazeRegionFeature = psi(ts.sample.x, gtGazeRegions.get(cnt));
			for (int i =0; i<dim;i++){
				gazePsi[i] += (laiFeature[i]-gtGazeRegionFeature[i])/this.K;
			}
		}
		return gazePsi;
	}
	
	@Override
	public double[] loss(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts) {
		//accumulate (average) classification loss for top K instances
		double v = 0;
		for (int cnt=0; cnt<this.K;cnt++){
			v += valueOf(ts.sample.x, ts.sample.hlist.get(cnt))/this.K;
		}
		Object[] lai = LAI(ts);
		double laiValue = (double)lai[1];
		ArrayList<Integer> gtGazeRegions = groundTruthGazeMap.get(ts.sample.x.getName());
		for (int cnt=0; cnt<this.K;cnt++){
			laiValue -= (valueOf(ts.sample.x, gtGazeRegions.get(cnt))/this.K);			
		}

		double[] lossTerm = new double[2];
		lossTerm[0] = ts.label == 1?(Math.max(1, v) - v):(Math.max(0, 1 + v));
		lossTerm[1] = ts.label == 1?(postradeoff * laiValue):(negtradeoff * laiValue);
		return lossTerm;

	}
//	
	@Override
	public double getLoss(List<TrainingSample<LatentRepresentationTopK<BagImage, Integer>>> l) {
		return (Double)null;
	}

	
	@Override
	public double getLoss(List<TrainingSample<LatentRepresentationTopK<BagImage, Integer>>> l, BufferedWriter trainingDetailFileOut) {
		double loss = 0;
		double classfication_loss = 0;
		double positive_gaze_loss_bound = 0;
		double negative_gaze_loss_bound = 0;
		double gaze_loss_bound = 0;
		double positive_gaze_loss = 0;
		double negative_gaze_loss = 0;
		for(TrainingSample<LatentRepresentationTopK<BagImage, Integer>> ts : l) {
			double[] example_loss = loss(ts);
			gaze_loss_bound += example_loss[1];
			if (ts.label == 1){
				positive_gaze_loss_bound+=example_loss[1];
				for (int cnt=0; cnt<this.K;cnt++){
					positive_gaze_loss+=1*getPositiveGazeLoss(ts, ts.sample.hlist.get(cnt));
				}
				positive_gaze_loss/=this.K;
//				loss += DoubleStream.of(example_loss).sum() ;
				loss += DoubleStream.of(example_loss).sum() /nb[0];
				classfication_loss +=example_loss[0]/nb[0];

			}
			if (ts.label == -1){
				negative_gaze_loss_bound+=example_loss[1];
				for (int cnt=0; cnt<this.K;cnt++){
					negative_gaze_loss+=1*getNegativeGazeLoss(ts, ts.sample.hlist.get(cnt));
				}
				negative_gaze_loss/=this.K;
//				loss += DoubleStream.of(example_loss).sum() ;
				loss += DoubleStream.of(example_loss).sum() /nb[1];
				classfication_loss +=example_loss[0]/nb[1];

			}
		}
		System.out.format("classification loss:%f, positive_gaze_loss_bound: %f, negative_gaze_loss_bound: %f,"
						+ " gaze_loss_bound:%f, positive_gaze_loss:%f, negative_gaze_loss:%f, gaze_loss:%f",
						classfication_loss, positive_gaze_loss_bound, negative_gaze_loss_bound, 
						gaze_loss_bound, positive_gaze_loss, negative_gaze_loss, positive_gaze_loss+negative_gaze_loss);
		try {
			trainingDetailFileOut.write("classification_loss:"+classfication_loss+
										" positive_gaze_loss_bound:"+positive_gaze_loss_bound+
										" negative_gaze_loss_bound:"+negative_gaze_loss_bound+
										" gaze_loss_bound:"+gaze_loss_bound+
										" positive_gaze_loss:"+positive_gaze_loss+
										" negative_gaze_loss:"+negative_gaze_loss+
										" gaze_loss:"+(positive_gaze_loss+negative_gaze_loss));
			trainingDetailFileOut.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		loss /= l.size();
		loss /=2;
		return loss;
	}
	
	public double testAP(List<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>> l) {
		List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
		for(int i=0; i<l.size(); i++) {
        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
			double score = 0;
			for (int cnt=0; cnt<this.K;cnt++){
				score += valueOf(l.get(i).sample.x,l.get(i).sample.hlist.get(cnt))/this.K;
			}
        	// label is changed to -1 1.
        	eval.add(new Pair<Integer,Double>((l.get(i).label), score)); //
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}
	
	public double testAPRegion(List<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>> l, File resFile) {
		List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
			for(int i=0; i<l.size(); i++) {
				double score=0;
				for (int cnt=0; cnt<this.K;cnt++){
					score += valueOf(l.get(i).sample.x,l.get(i).sample.hlist.get(cnt))/this.K;
				}
				Integer yi = l.get(i).label;
				Integer yp = score > 0 ? 1 : -1;
				out.write(Double.valueOf(score) + ","+Integer.valueOf(yp) +","+Integer.valueOf(yi) +","+ l.get(i).sample.hlist.toString()+","+groundTruthGazeMap.get(l.get(i).sample.x.getName()).toString()+","+l.get(i).sample.x.getName()+"\n");
				out.flush();
				eval.add(new Pair<Integer,Double>((l.get(i).label), score)); //
			}
			out.close();
	        	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}

	public void testScoreRegion(List<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>> l, File resFile) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
			for(int i=0; i<l.size(); i++) {
				double score=0;
				for (int cnt=0; cnt<this.K;cnt++){
					score += valueOf(l.get(i).sample.x,l.get(i).sample.hlist.get(cnt))/this.K;
				}
				Integer yi = 0;
				Integer yp = score > 0 ? 1 : -1;
				out.write(Double.valueOf(score) + ","+Integer.valueOf(yp) +","+Integer.valueOf(yi) +","+ l.get(i).sample.hlist.toString()+","+"x"+","+l.get(i).sample.x.getName()+"\n");
				out.flush();
			}
			out.close();
	        	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}


}
