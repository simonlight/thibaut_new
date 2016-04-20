/**
 * 
 */
package fr.durandt.jstruct.ssvm.ranking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.ssvm.SSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.Pair;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class DoubleRankAPSSVMPegasos extends SSVMPegasos<RankingInput,RankingOutput> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6439820921646750434L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected RankingOutput lossAugmentedInference(STrainingSample<RankingInput, RankingOutput> ts, double[] w) {

		List<Pair<Integer,Double>> positiveExamples = new ArrayList<Pair<Integer,Double>>(ts.input.getNpos());
		List<Pair<Integer,Double>> negativeExamples = new ArrayList<Pair<Integer,Double>>(ts.input.getNneg());

		for(int i=0; i<ts.input.getNumberOfExamples(); i++) {
			double score = linear.valueOf(w, ts.input.getFeature(i));

			if(ts.input.getLabel(i) == 1) {
				positiveExamples.add(new Pair<Integer, Double>(i, score));
			}
			else {
				negativeExamples.add(new Pair<Integer, Double>(i, score));
			}
		}

		// sort positiveExamples and negativeExamples in descending order of score
		Collections.sort(positiveExamples, Collections.reverseOrder());
		Collections.sort(negativeExamples, Collections.reverseOrder());

		int negativeId = 0;
		int positiveId = 0; 
		Integer[] imgIndexMap = new Integer[ts.input.getNumberOfExamples()];
		for(int i=0; i<ts.input.getNumberOfExamples(); i++){
			if(ts.input.getLabel(i) == 1){
				imgIndexMap[positiveExamples.get(positiveId).getKey()] = positiveId;
				positiveId++;
			}
			else{
				imgIndexMap[negativeExamples.get(negativeId).getKey()] = negativeId;
				negativeId++;
			}
		}    

		RankingOutput rankLAI = findOptimumNegLocations(ts.input, positiveExamples, negativeExamples, imgIndexMap);
		return rankLAI;
	}

	protected RankingOutput findOptimumNegLocations(RankingInput x, List<Pair<Integer,Double>> positiveExamples, 
			List<Pair<Integer,Double>> negativeExamples, Integer[] imgIndexMap) {

		double maxValue = 0;
		double currentValue = 0;
		int maxIndex = -1;

		Integer[] optimumLocNegImg = new Integer[x.getNneg()];

		// for every jth negative image
		for(int j=1; j<=x.getNneg(); j++){
			maxValue = 0;
			maxIndex = x.getNpos()+1;
			// k is what we are maximising over. There would be one k_max for each negative image j
			currentValue = 0;
			for(int k=x.getNpos(); k>=1; k--){
				currentValue += (1/(double)x.getNpos())*((j/(double)(j+k))-((j-1)/(double)(j+k-1))) 
						- (2/(double)(x.getNpos()*x.getNneg()))*(positiveExamples.get(k-1).getValue() - negativeExamples.get(j-1).getValue());
				if(currentValue > maxValue){
					maxValue = currentValue;
					maxIndex = k;
				}
			}
			optimumLocNegImg[j-1] = maxIndex;
		}

		return encodeRanking(x, positiveExamples, negativeExamples, imgIndexMap, optimumLocNegImg);
	}

	protected RankingOutput encodeRanking(RankingInput x, List<Pair<Integer,Double>> positiveExamples, 
			List<Pair<Integer,Double>> negativeExamples, Integer[] imgIndexMap, Integer[] optimumLocNegImg){

		Integer[] ranking = new Integer[x.getNumberOfExamples()];
		for(int i=0; i<ranking.length; i++) {
			ranking[i] = 0;
		}

		for(int i=0; i<x.getNumberOfExamples(); i++){
			for(int j=i+1; j<x.getNumberOfExamples(); j++){        
				if(i == j){
					// Nothing to do
				}
				else if(x.getLabel(i) == x.getLabel(j)){                
					if(x.getLabel(i) == 1){                                 
						if(positiveExamples.get(imgIndexMap[i]).getValue() > positiveExamples.get(imgIndexMap[j]).getValue()){
							ranking[i]++;
							ranking[j]--;
						}
						else if(positiveExamples.get(imgIndexMap[j]).getValue() > positiveExamples.get(imgIndexMap[i]).getValue()){
							ranking[i]--;
							ranking[j]++;
						}
						else{
							if(i < j){
								ranking[i]++;
								ranking[j]--;
							}
							else{
								ranking[i]--;
								ranking[j]++;
							}
						}
					}
					else{
						if(negativeExamples.get(imgIndexMap[i]).getValue() > negativeExamples.get(imgIndexMap[j]).getValue()){
							ranking[i]++;
							ranking[j]--;
						}
						else if(negativeExamples.get(imgIndexMap[j]).getValue() > negativeExamples.get(imgIndexMap[i]).getValue()){
							ranking[i]--;
							ranking[j]++;
						}
						else{
							if(i < j){
								ranking[i]++;
								ranking[j]--;
							}
							else{
								ranking[i]--;
								ranking[j]++;
							}
						}
					}        
				}
				else if((x.getLabel(i) == 1) && (x.getLabel(j) == 0)){
					int iPrime = imgIndexMap[i]+1;
					int jPrime = imgIndexMap[j]+1;
					int ojPrime = optimumLocNegImg[jPrime-1];

					if((ojPrime - iPrime - 0.5) > 0){
						ranking[i]++;
						ranking[j]--;
					}
					else{
						ranking[i]--;
						ranking[j]++;
					}
				}
				else if((x.getLabel(i) == 0) && (x.getLabel(j) == 1)){
					int iPrime = imgIndexMap[i]+1;
					int jPrime = imgIndexMap[j]+1;
					int oiPrime = optimumLocNegImg[iPrime-1];

					if((jPrime - oiPrime + 0.5) > 0){
						ranking[i]++;
						ranking[j]--;
					}
					else{
						ranking[i]--;
						ranking[j]++;
					}
				}                    
			}        
		}    

		return new RankingOutput(ranking, x.getNpos(), x.getNneg());
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#prediction(java.lang.Object, double[])
	 */
	@Override
	protected RankingOutput prediction(RankingInput x, double[] w) {
		
		List<Pair<Integer,Double>> examples = new ArrayList<Pair<Integer,Double>>(x.getNumberOfExamples());

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			double score = linear.valueOf(w, x.getFeature(i));
			examples.add(new Pair<Integer, Double>(i, score));
		}

		// sort examples in descending order of score
		Collections.sort(examples, Collections.reverseOrder());
		Integer[] ranking = new Integer[examples.size()];
		
		for(int i=0; i<examples.size(); i++) {
			ranking[examples.get(i).getKey()] = x.getNumberOfExamples() - i;
		}
		
		RankingOutput prediction = new RankingOutput(ranking, x.getNpos(), x.getNneg());
		return prediction;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#delta(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(RankingOutput yi, RankingOutput y) {
		double delta = 1 - RankingOutput.averagePrecision(yi, y);
		return delta;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#psi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(RankingInput x, RankingOutput y) {
		double[] psi = new double[dim];

		int[] count = new int[x.getNumberOfExamples()];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			if(x.getLabel(i) == 1) {
				for(int j=0; j<x.getNumberOfExamples(); j++) {
					if(x.getLabel(j) == 0) {
						int yij = -1;
						if(y.getRanking(i) > y.getRanking(j)){
							yij = 1;
						}
						if(yij == 1) { // Yij = +1
							count[i]++;
							count[j]--;
						}
						else {	// Yij = -1
							count[i]--;
							count[j]++;
						}
					}
				}
			}
		}
		
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			double[] psii = x.getFeature(i);
			for(int d=0; d<dim; d++) {
				psi[d] += psii[d] * count[i];
			}
		}

		// Divide by the number of relevant examples * number of irrelevant examples
		double c = x.getNpos()*x.getNneg();
		for(int d=0; d<dim; d++) {
			psi[d] /= c;
		}

		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<RankingInput, RankingOutput>> l) {
		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank AP SSVM \t P= " + l.get(0).input.getNpos() + "\t N= " + l.get(0).input.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.getFeature(0).length;

		// Initialize w
		w = new double[dim];
		//w[0] = 1;
	}
	
	/**
	 * Compute the average precision
	 * @param l
	 * @return
	 */
	public double averagePrecision(List<STrainingSample<RankingInput, RankingOutput>> l) {
		// Ranking prediction
		RankingOutput prediction = prediction(l.get(0).input);
		// Compute the average precision with the predicted ranking
		return RankingOutput.averagePrecision(l.get(0).output, prediction);
	}
}
