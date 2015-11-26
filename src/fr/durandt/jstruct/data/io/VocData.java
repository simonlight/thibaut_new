/**
 * 
 */
package fr.durandt.jstruct.data.io;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class VocData {

	private String name = null;
	
	private int indexRegionAction = 0;
	
	private double scoreAction = 0;
	
	private int label = 0;
	
	
	public void print() {
		System.out.println("----------------------------------------------------------------");
		System.out.println("name= " + name);
		System.out.println("label= " + label);
		System.out.println("action - score= " + scoreAction);
		System.out.println("action - index region= " + indexRegionAction);
		System.out.println("----------------------------------------------------------------");
	}
	

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return the indexRegionAction
	 */
	public int getIndexRegionAction() {
		return indexRegionAction;
	}

	/**
	 * @param indexRegionAction the indexRegionAction to set
	 */
	public void setIndexRegionAction(int indexRegionAction) {
		this.indexRegionAction = indexRegionAction;
	}

	/**
	 * @return the scoreAction
	 */
	public double getScoreAction() {
		return scoreAction;
	}

	/**
	 * @param scoreAction the scoreAction to set
	 */
	public void setScoreAction(double scoreAction) {
		this.scoreAction = scoreAction;
	}

	/**
	 * @return the label
	 */
	public int getLabel() {
		return label;
	}

	/**
	 * @param label the label to set
	 */
	public void setLabel(int label) {
		this.label = label;
	}
	
	
	
	
}
