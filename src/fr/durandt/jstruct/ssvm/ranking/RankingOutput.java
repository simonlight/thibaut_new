/**
 * 
 */
package fr.durandt.jstruct.ssvm.ranking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.Pair;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class RankingOutput {

	private Integer[] ranking = null;
	private Integer[] labels = null;

	private List<Double> compressedRepresentation = null;

	private int nPos;
	private int nNeg;

	private Integer[][] y = null;

	public RankingOutput() {

		nPos = 0;
		nNeg = 0;

		labels = null;
		ranking = null;
	}

	public RankingOutput(int nPos, int nNeg) {

		this.nPos = nPos;
		this.nNeg = nNeg;

		labels = null;
		ranking = null;
	}


	public RankingOutput(List<STrainingSample<double[], Integer>> examples) {
		// Initialize labels
		labels = new Integer[examples.size()];
		for(int i=0; i<examples.size(); i++) {
			labels[i] = examples.get(i).output;
			if(examples.get(i).output == 1) {
				nPos++;
			}
			else {
				nNeg++;
			}
		}

		// Initialize ranking
		ranking = new Integer[labels.length];
		for(int i=0; i<ranking.length; i++) {
			ranking[i] = 0;
		}

		for(int i=0; i<labels.length; i++) {
			for(int j=i+1; j<labels.length; j++){
				if(labels[i] == 1){
					if(labels[j] != 0){
						ranking[i]++;
						ranking[j]--;
					}              
				}
				else{
					if(labels[j] == 1){
						ranking[i]--;
						ranking[j]++;
					}
				}
			}
		}
	}

	public void initialize(List<Integer> labelsGT) {
		// Initialize labels
		labels = new Integer[labelsGT.size()];
		for(int i=0; i<labelsGT.size(); i++) {
			labels[i] = labelsGT.get(i);
			if(labelsGT.get(i) == 1) {
				nPos++;
			}
			else {
				nNeg++;
			}
		}

		// Initialize ranking
		ranking = new Integer[labels.length];
		for(int i=0; i<ranking.length; i++) {
			ranking[i] = 0;
		}

		for(int i=0; i<labels.length; i++){
			for(int j=i+1; j<labels.length; j++){
				if(labels[i] == 1){
					if(labels[j] != 0){
						ranking[i]++;
						ranking[j]--;
					}              
				}
				else{
					if(labels[j] == 1){
						ranking[i]--;
						ranking[j]++;
					}
				}
			}
		}
	}

	public RankingOutput(Integer[] ranking, int nPos, int nNeg) {

		this.ranking = ranking;
		this.nPos = nPos;
		this.nNeg = nNeg;

		labels = new Integer[this.ranking.length];
	}

	public RankingOutput(Integer[] ranking, Integer[] labels, int nPos, int nNeg) {

		this.ranking = ranking;
		this.nPos = nPos;
		this.nNeg = nNeg;
		this.labels = labels;
	}

	public static double averagePrecision(RankingOutput yi, RankingOutput y) {

		//yi.printInfo();
		//y.printInfo();

		// Stores rank of all images
		int[] ranking = new int[yi.getNumberOfExamples()]; 
		// Stores list of images sorted by rank. Higher rank to lower rank 
		int[] sortedExamples = new int[yi.getNumberOfExamples()]; 

		// convert rank matrix to rank list
		for(int i=0; i<yi.getNumberOfExamples(); i++){
			// start with lowest rank for each sample i.e 1 
			ranking[i] = 1; 
			for(int j=0; j<yi.getNumberOfExamples(); j++){
				if(y.getRanking(i) > y.getRanking(j)){
					ranking[i] = ranking[i] + 1;
				} 
			}
			sortedExamples[yi.getNumberOfExamples() - ranking[i]] = i;
		}  

		int posCount = 0;
		int totalCount = 0;
		double precisionAti = 0;
		for(int i=0; i<yi.getNumberOfExamples(); i++){
			int label = yi.getLabel(sortedExamples[i]);
			if(label == 1){
				posCount++;
				totalCount++;
			}
			else{
				totalCount++;
			}
			if(label == 1){
				precisionAti = precisionAti + (double)posCount/(double)totalCount;
			}
		}
		precisionAti = precisionAti/posCount;

		return precisionAti;
	}

	public static double averagePrecisionY(RankingOutput yi, RankingOutput y) {

		yi.printInfo();
		y.printInfo();

		// Stores rank of all images
		int[] ranking = new int[yi.getNumberOfExamples()]; 
		// Stores list of images sorted by rank. Higher rank to lower rank 
		int[] sortedExamples = new int[yi.getNumberOfExamples()]; 

		// Convert rank matrix to rank list
		for(int i=0; i<yi.getNumberOfExamples(); i++){
			// Start with lowest rank for each sample i.e 1 
			ranking[i] = 1; 
			for(int j=0; j<yi.getNumberOfExamples(); j++){
				if(y.getY(i, j) == 1){
					ranking[i] = ranking[i] + 1;
				}
			}
			sortedExamples[yi.getNumberOfExamples() - ranking[i]] = i;
		}  

		int posCount = 0;
		int totalCount = 0;
		double precisionAti = 0;
		for(int i=0; i<yi.getNumberOfExamples(); i++){
			int label = yi.getLabel(sortedExamples[i]);
			if(label == 1){
				posCount++;
				totalCount++;
			}
			else{
				totalCount++;
			}
			if(label == 1){
				precisionAti = precisionAti + (double)posCount/(double)totalCount;
			}
		}
		precisionAti = precisionAti/posCount;

		return precisionAti;
	}

	/**
	 * Compute the area under the ROC curve
	 * @param yi the ground truth
	 * @param y	the predicted ranking
	 * @return
	 */
	public static double rocArea(RankingOutput yi, RankingOutput y) {
		List<Pair<Integer,Double>> scores = new ArrayList<Pair<Integer,Double>>();
		for(int i=0; i<yi.getNumberOfExamples(); i++) {
			scores.add(new Pair<Integer,Double>(i, (double)y.getLabel(i)));
		}
		Collections.sort(scores, Collections.reverseOrder());

		int numn = 0;
		int nump = 0;
		int swappedPairs=0;
		for(int i=0; i<yi.getNumberOfExamples(); i++) {
			if(yi.getLabel(scores.get(i).getKey()) == 1) {
				swappedPairs += numn;
				nump++;
			}
			else {
				numn++;
			}
		}
		//return 100.0-100.0*swappedPairs/((double)numn)/((double)nump);
		return 1.0 - 1.0*swappedPairs/(numn)/(nump);
	}

	/**
	 * Compute the area under the ROC curve with the ranking matrix y
	 * @param yi the ground truth
	 * @param y	the predicted ranking
	 * @return
	 */
	public static double rocAreaY(RankingOutput yi, RankingOutput y) {

		double swappedPair = 0;
		for(int i=0; i<yi.getNumberOfExamples(); i++) {
			for(int j=0; j<yi.getNumberOfExamples(); j++) {
				if(yi.getY(i, j) == 1) 
					swappedPair += yi.getY(i, j) - y.getY(i, j);
			}
		}
		return 1 - swappedPair * 0.5 / (yi.getnPos() * yi.getnNeg());
	}

	public void initializeCompressedRepresentation(String type) {
		compressedRepresentation = new ArrayList<Double>();
		if(type.compareToIgnoreCase("ROC") == 0) {
			for(int i=0; i<getNumberOfExamples(); i++) {
				if(labels[i] == 1) {
					compressedRepresentation.add(i, 50.0/nPos);
					//compressedRepresentation.add(i, 0.5/nPos);
				}
				else {
					compressedRepresentation.add(i, -50.0/nNeg);
					//compressedRepresentation.add(i, -0.5/nNeg);
				}
			}
		}
		else if(type.compareToIgnoreCase("AP") == 0) {
			for(int i=0; i<getNumberOfExamples(); i++) {
				if(labels[i] == 1) {
					compressedRepresentation.add(i, (double)nNeg);
				}
				else {
					compressedRepresentation.add(i, (double)-nPos);
				}
			}
		}
	}

	/*public void computeCompressedRepresentation(String type) {
		compressedRepresentation = new ArrayList<Double>();
		if(type.compareToIgnoreCase("ROC") == 0) {
			for(int i=0; i<getNumberOfExamples(); i++) {
				compressedRepresentation.add(i, labels[i]*50.0/(nPos*nNeg));
			}
		}
	}*/

	public void computeYwithLabels() {
		y = new Integer[getNumberOfExamples()][getNumberOfExamples()];
		for(int i=0; i<getNumberOfExamples(); i++) {
			for(int j=i; j<getNumberOfExamples(); j++) {
				if(getLabel(i) > getLabel(j)) {
					y[i][j] = 1;
					y[j][i] = -1;
				}
				else if(getLabel(i) < getLabel(j)) {
					y[i][j] = -1;
					y[j][i] = 1;
				}
				else {
					y[i][j] = 0;
					y[j][i] = 0;
				}
			}
		}
	}

	public void printInfo() {
		System.out.println("Ranking output \t nPos= " + nPos + "\tnNeg= " + nNeg);
		if(labels != null) {
			System.out.println("labels (" + labels.length + ")\t"+ Arrays.toString(labels));
		}
		else {
			System.out.println("labels null");
		}

		if(ranking != null) {
			System.out.println("ranking (" + ranking.length + ")\t"+ Arrays.toString(ranking));
		}
		else {
			System.out.println("ranking null");
		}

		if(compressedRepresentation != null) {
			System.out.println("compressed representation (" + compressedRepresentation.size() + ")\t" + compressedRepresentation);
		}
		else {
			System.out.println("compressed representation null");
		}

		if(y != null) {
			System.out.println("y (" + y.length + " * " + y[0].length + ")");
		}
		else {
			System.out.println("y null");
		}
	}

	public void printY() {
		for(int i=0; i<getNumberOfExamples(); i++) {
			for(int j=0; j<getNumberOfExamples(); j++) {
				System.out.print(y[i][j] + "\t");
			}
			System.out.println();
		}
	}

	/**
	 * @return the ranking
	 */
	public Integer[] getRanking() {
		return ranking;
	}

	public Integer getRanking(int i) {
		return ranking[i];
	}

	/**
	 * @param ranking the ranking to set
	 */
	public void setRanking(Integer[] ranking) {
		this.ranking = ranking;
	}

	/**
	 * @return the labels
	 */
	public Integer[] getLabels() {
		return labels;
	}

	public Integer getLabel(int i) {
		return labels[i];
	}

	/**
	 * @param labels the labels to set
	 */
	public void setLabels(Integer[] labels) {
		this.labels = labels;
	}

	/**
	 * @return
	 */
	public int getNumberOfExamples() {
		return (nPos + nNeg);
	}

	/**
	 * @return the nPos
	 */
	public int getnPos() {
		return nPos;
	}

	/**
	 * @param nPos the nPos to set
	 */
	public void setnPos(int nPos) {
		this.nPos = nPos;
	}

	/**
	 * @return the nNeg
	 */
	public int getnNeg() {
		return nNeg;
	}

	/**
	 * @param nNeg the nNeg to set
	 */
	public void setnNeg(int nNeg) {
		this.nNeg = nNeg;
	}

	/**
	 * @return the compressedRepresentation
	 */
	public List<Double> getCompressedRepresentation() {
		return compressedRepresentation;
	}

	public double getCompressedRepresentation(int i) {
		return compressedRepresentation.get(i);
	}

	/**
	 * @param compressedRepresentation the compressedRepresentation to set
	 */
	public void setCompressedRepresentation(List<Double> compressedRepresentation) {
		this.compressedRepresentation = compressedRepresentation;
	}

	public void setCompressedRepresentation(int i, double val) {
		this.compressedRepresentation.set(i, val);
	}

	/**
	 * @return the y
	 */
	public Integer[][] getY() {
		return y;
	}

	public Integer getY(int i, int j) {
		return y[i][j];
	}

	/**
	 * @param y the y to set
	 */
	public void setY(Integer[][] y) {
		this.y = y;
	}


}
