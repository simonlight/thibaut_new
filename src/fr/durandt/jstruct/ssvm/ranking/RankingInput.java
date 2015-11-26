/**
 * 
 */
package fr.durandt.jstruct.ssvm.ranking;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class RankingInput {

	/**
	 * Number of positives examples
	 */
	private int npos;

	/**
	 * Number of negatives examples
	 */
	private int nneg;

	/**
	 * Features
	 */
	private List<double[]> features = null;
	
	private List<Integer> labels = null;

	public RankingInput(List<STrainingSample<double[], Integer>> examples) {

		features = new ArrayList<double[]>(examples.size());
		labels = new ArrayList<Integer>(examples.size());
		for(int i=0; i<examples.size(); i++) {
			features.add(i, examples.get(i).input.clone());
			labels.add(i, examples.get(i).output);
			if(examples.get(i).output == 1) {
				npos++;
			}
			else {
				nneg++;
			}
		}
		//System.out.println("features " + features.size() + "\tnpos= " + npos + "\tnneg= " + nneg);
		//System.out.println("labels (" + labels.size() + ")\t"+ labels);
	}

	/**
	 * @return the npos
	 */
	public int getNpos() {
		return npos;
	}

	/**
	 * @param npos the npos to set
	 */
	public void setNpos(int npos) {
		this.npos = npos;
	}

	/**
	 * @return the nneg
	 */
	public int getNneg() {
		return nneg;
	}

	/**
	 * @param nneg the nneg to set
	 */
	public void setNneg(int nneg) {
		this.nneg = nneg;
	}

	/**
	 * @return the features
	 */
	public List<double[]> getFeatures() {
		return features;
	}

	/**
	 * @return the i-th feature
	 */
	public double[] getFeature(int i) {
		return features.get(i);
	}

	/**
	 * @param features the features to set
	 */
	public void setFeatures(List<double[]> features) {
		this.features = features;
	}

	/**
	 * @return
	 */
	public int getNumberOfExamples() {
		return (npos + nneg);
	}

	/**
	 * @return the labels
	 */
	public List<Integer> getLabels() {
		return labels;
	}

	/**
	 * @param labels the labels to set
	 */
	public void setLabels(List<Integer> labels) {
		this.labels = labels;
	}

	/**
	 * @return 
	 */
	public Integer getLabel(int i) {
		return labels.get(i);
	}
	
}
