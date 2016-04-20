/**
 * 
 */
package fr.durandt.jstruct.variable;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class BagLabel extends Bag {

	// List of labels for each instance
	protected List<Integer> labels = null;
	protected double labelProportion = 0;
	
	public BagLabel() {
		super();
		labels = new ArrayList<Integer>();
	}
	
	public BagLabel(Bag bag) {
		name = bag.getName();
		instances = bag.getInstances();
		labels = new ArrayList<Integer>();
		for(int i=0; i<instances.size(); i++) {
			labels.add(0);
		}
	}

	public void addInstance(double[] instance, int label) {
		super.addInstance(instance);
		addLabel(label);
	}

	public void addInstance(int index, double[] instance, int label) {
		super.addInstance(index, instance);
		addLabel(index, label);
	}
	
	public void addLabel(int label) {
		labels.add(label);
		// Update the label proportion
		computeLabelProportion();
	}
	
	public void addLabel(int index, int label) {
		labels.add(index, label);
		// Update the label proportion
		computeLabelProportion();
	}
	
	public int getLabel(int index) {
		return labels.get(index);
	}
	
	/**
	 * Compute the proportion of positive labels
	 */
	protected void computeLabelProportion() {
		double proportion = 0;
		for(Integer label : labels) {
			if(label == 1) {
				proportion += 1;
			}
		}
		labelProportion = proportion/labels.size();
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
	
	public double getLabelProportion() {
		return labelProportion;
	}

	/**
	 * @param labelProportion the labelProportion to set
	 */
	public void setLabelProportion(double labelProportion) {
		this.labelProportion = labelProportion;
	}
	
}
