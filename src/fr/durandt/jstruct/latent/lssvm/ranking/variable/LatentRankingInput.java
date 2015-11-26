/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.ranking.variable;

import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LatentRankingInput<X,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Number of positives examples
	 */
	protected int npos;

	/**
	 * Number of negatives examples
	 */
	protected int nneg;

	/**
	 * Label of each example
	 */
	protected List<Integer> labels = null;

	/**
	 * List of examples
	 */
	protected List<LatentRepresentation<X,H>> examples = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Abstract methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	public abstract double[] getFeature(int i, H h);
	
	public void print() {
		System.out.println("pos= " + npos + "\tneg= " + nneg);
		System.out.println("labels= " + labels);
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters & setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

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

	public Integer getLabel(int i) {
		return labels.get(i);
	}

	/**
	 * @return the number of examples
	 */
	public int getNumberOfExamples() {
		return examples.size();
	}

	public H getLatent(int i) {
		return examples.get(i).h;
	}
	
	public void setLatent(int i, H h) {
		examples.get(i).h = h;
	}
	
	public X getExample(int i) {
		return examples.get(i).x;
	}

}
