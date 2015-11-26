package fr.durandt.jstruct.variable;

import java.util.ArrayList;
import java.util.List;

/**
 * For weakly-supervised detection, the bag is the whole image and each instance represents an region of the image
 * 
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class BagImage extends Bag {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * List of file for each instance
	 */
	protected List<String> instanceFiles = null;

	/**
	 * Height of the image
	 */
	protected int height = -1;

	/**
	 * Width of the image
	 */
	protected int width = -1;

	/**
	 * Full path to the image
	 */
	protected String imageFile = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Constructor
	 */
	public BagImage() {
		super();
		instanceFiles = new ArrayList<String>();
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Add an instance feature with the name of the instance feature file
	 * @param instance
	 * @param fileInstance
	 */
	public void addInstance(double[] instance, String fileInstance) {
		super.addInstance(instance);
		addInstanceFile(fileInstance);
	}

	/**
	 * Add an instance feature with the name of the instance feature file at a given index
	 * @param index
	 * @param instance
	 * @param fileInstance
	 */
	public void addInstance(int index, double[] instance, String fileInstance) {
		super.addInstance(index, instance);
		addInstanceFile(index, fileInstance);
	}

	/**
	 * Add a new instance file 
	 * @param newInstanceFile
	 */
	public void addInstanceFile(String newInstanceFile) {
		instanceFiles.add(newInstanceFile);
	}

	/**
	 * Add a new instance file at position index
	 * @param index
	 * @param newInstanceFile
	 */
	public void addInstanceFile(int index, String newInstanceFile) {
		instanceFiles.add(index, newInstanceFile);
	}

	/**
	 * Return the index of the instance which instance file is targetInstanceFile
	 * @param fileInstanceTarget
	 * @return
	 */
	protected int getInstanceIndex(String targetInstanceFile) {
		for(int i=0; i<instanceFiles.size(); i++) {
			if(instanceFiles.get(i).compareTo(targetInstanceFile) == 0) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Remove the instance with the target instance file
	 * @param fileInstanceTarget
	 */
	public void removeInstance(String targetInstanceFile) {
		// find the index of the instance target
		int index = getInstanceIndex(targetInstanceFile);
		// remove the instance
		removeInstance(index);
	}

	/**
	 * Remove the index-th instance
	 * @param index
	 */
	public void removeInstance(int index) {
		if(index >= 0) {
			// remove the instance
			super.removeInstance(index);
			// remove the file name
			instanceFiles.remove(index);
		}
	}

	public String toString() {
		String s = super.toString() + "\theight: " + height + "\twidth: " + width;
		if(instanceFiles != null) {
			s += "\tinstanceFiles: " + instanceFiles.size();
		}
		return s;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Return the index-th instance file
	 * @param index
	 * @return instance file
	 */
	public String getInstanceFile(int index) {
		return instanceFiles.get(index);
	}

	/**
	 * Set a new index-th instance file
	 * @param index
	 * @param newFile
	 * @return
	 */
	public String setInstanceFile(int index, String newFile) {
		return instanceFiles.set(index, newFile);
	}

	/**
	 * @return the height of the image
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * @param height the height of the image to set
	 */
	public void setHeight(int height) {
		this.height = height;
	}

	/**
	 * @return the width of the image
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * @param width the width of the image to set
	 */
	public void setWidth(int width) {
		this.width = width;
	}

	/**
	 * Return the list of instance files
	 * @return
	 */
	public List<String> getInstanceFiles() {
		return instanceFiles;
	}

	/**
	 * Set the list of instance files
	 * @param instanceFiles
	 */
	public void setInstanceFiles(List<String> instanceFiles) {
		this.instanceFiles = instanceFiles;
	}

	/**
	 * @return the imageFile
	 */
	public String getImageFile() {
		return imageFile;
	}

	/**
	 * @param imageFile the imageFile to set
	 */
	public void setImageFile(String imageFile) {
		this.imageFile = imageFile;
	}
	
}
