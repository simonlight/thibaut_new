/**
 * 
 */
package fr.durandt.jstruct.variable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Bag for weakly-supervised detection
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class BagImageRegion extends BagImage {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * List of the position of each region
	 */
	protected List<Integer[]> regions = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Constructor
	 */
	public BagImageRegion() {
		super();
		regions = new ArrayList<Integer[]>();
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Add new region
	 * @param region new region
	 */
	public void addRegion(Integer[] region) {
		regions.add(region);
	}

	/**
	 * Add new region at position index
	 * @param index
	 * @param region
	 */
	public void addRegion(int index, Integer[] region) {
		regions.add(index, region);
	}

	/**
	 * Add new instance, instance file and region
	 * @param instance
	 * @param fileInstance
	 * @param region
	 */
	public void addInstance(double[] instance, String fileInstance, Integer[] region) {
		super.addInstance(instance, fileInstance);
		addRegion(region);
	}

	/**
	 * Add new instance, instance file and region at position index
	 * @param index
	 * @param instance
	 * @param fileInstance
	 * @param region
	 */
	public void addInstance(int index, double[] instance, String fileInstance, Integer[] region) {
		super.addInstance(index, instance, fileInstance);
		addRegion(index, region);
	}

	public void removeOverlap(Integer[] region, double overlapMax) {
		for(int i=numberOfInstances()-1; i>=0; i--) {
			double overlap = computeOverlap(regions.get(i),region);
			//System.out.println(Arrays.toString(regions.get(i)) + "\t" + Arrays.toString(region));
			//System.out.println("overlap= " + overlap);

			if(overlap > overlapMax) {
				removeInstance(i);
			}
		}
	}

	/**
	 * Compute overlap between region b1 and b2
	 * @param b1
	 * @param b2
	 * @return overlap between region b1 and b2
	 */
	protected double computeOverlap(Integer[] b1, Integer[] b2) {
		if(b1.length != 4 || b2.length != 4) {
			System.out.println("b1= " + Arrays.toString(b1));
			System.out.println("b2= " + Arrays.toString(b2));
			System.exit(0);
		}

		int[] bi = {Math.max(b1[0],b2[0]), Math.max(b1[1],b2[1]), Math.min(b1[2],b2[2]), Math.min(b1[3],b2[3])};
		int iw = bi[2]-bi[0]+1;
		int ih = bi[3]-bi[1]+1;

		double ov = 0;
		if(iw>0 && ih>0) {
			double ua=(b1[2]-b1[0]+1)*(b1[3]-b1[1]+1) + (b2[2]-b2[0]+1)*(b2[3]-b2[1]+1)-iw*ih;
			ov=iw*ih/ua;
		}

		return ov;
	}

	/**
	 * Remove the instance with the target instance file
	 * @param fileInstanceTarget
	 */
	@Override
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
	@Override
	public void removeInstance(int index) {
		if(index >= 0) {
			// remove index-th instance
			super.removeInstance(index);
			// remove index-th region
			regions.remove(index);
		}
	}

	@Override
	public String toString() {
		String s = super.toString();
		if(instanceFiles != null) {
			s += "\tregions: " + regions.size();
		}
		return s;
	}

	@Override
	public void print() {
		System.out.println(toString());
		System.out.println("image file: " + imageFile);
		if(bagFeature != null) {
			System.out.println("bag feature dim= " + bagFeature.length);
		}
		int nbInstances = 0;
		if(instances != null) {
			nbInstances = Math.max(nbInstances, instances.size());
		}
		if(instanceFiles != null) {
			nbInstances = Math.max(nbInstances, instanceFiles.size());
		}
		if(regions != null) {
			nbInstances = Math.max(nbInstances, regions.size());
		}
		for(int i=0; i<nbInstances; i++) {
			System.out.print(i + " - ");
			if(instanceFiles.size() > i) {
				System.out.print(instanceFiles.get(i) + "\t");
			}
			if(instances.size() > i) {
				System.out.print(instances.get(i).length + "\t");
			}
			if(regions.size() > i) {
				System.out.print(Arrays.toString(regions.get(i)) + "\t");
			}
			System.out.println();
		}
	}

	@Override
	public void readXMLFile(File file, int dim, int verbose) {

		if(file.exists()) {

			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

			try {
				DocumentBuilder builder = factory.newDocumentBuilder();
				Document document = builder.parse(file);

				Element racine = document.getDocumentElement();
				NodeList racineNoeuds = racine.getChildNodes();
				int nbRacineNoeuds = racineNoeuds.getLength();

				// Number of instances
				int numberOfInstances = -1;

				for(int i=0; i<nbRacineNoeuds; i++) {
					if(racineNoeuds.item(i).getNodeType() == Node.ELEMENT_NODE) {
						Element e = (Element) racineNoeuds.item(i);
						if(verbose > 2) System.out.println(i + "\t" + e + "\t" + e.getTextContent());

						if(e.getTagName().compareToIgnoreCase("name") == 0) {
							// Read the name of the bag
							String name = e.getTextContent();
							setName(name);
						}
						else if(e.getTagName().compareToIgnoreCase("numberofinstances") == 0) {
							// Read the number of instances
							numberOfInstances = Integer.parseInt(e.getTextContent());
						}
						else if(e.getTagName().compareToIgnoreCase("feature") == 0) {
							// Read the global feature representation of the bag
							int featureDim = Integer.parseInt(e.getAttribute("dim"));
							if(dim >= 0 && featureDim != dim) {
								System.out.println("ERROR - read dimension feature " + featureDim + " vs " + dim);
								System.exit(0);
							}
							String stringFeature = e.getTextContent();
							setBagFeature(readFeature(stringFeature, featureDim));
						}
						else if(e.getTagName().compareToIgnoreCase("imagefile") == 0) {
							// Read Full path to the image
							setImageFile(e.getTextContent());
						}
						else if(e.getTagName().compareToIgnoreCase("width") == 0) {
							// Read the width of the image
							setWidth(Integer.parseInt(e.getTextContent()));
						}
						else if(e.getTagName().compareToIgnoreCase("height") == 0) {
							// Read the height of the image
							setHeight(Integer.parseInt(e.getTextContent()));
						}
						else if(e.getTagName().compareToIgnoreCase("instance") == 0) {
							// Read an instance

							NodeList childNoeuds = e.getChildNodes();
							int nbChildNoeuds = childNoeuds.getLength();

							// Index of instance
							int index = -1;

							// region
							int x1 = -1;
							int y1 = -1;
							int x2 = -1;
							int y2 = -1;

							// Instance feature
							double[] feature = null;

							for(int j=0; j<nbChildNoeuds; j++) {
								if(childNoeuds.item(j).getNodeType() == Node.ELEMENT_NODE) {
									Element childElement = (Element) childNoeuds.item(j);

									if(verbose > 2) System.out.println(j + "\t" + childElement);

									if(childElement.getTagName().compareToIgnoreCase("index") == 0) {
										// Read index of instance
										index = Integer.parseInt(childElement.getTextContent());
									}
									else if(childElement.getTagName().compareToIgnoreCase("x1") == 0) {
										x1 = Integer.parseInt(childElement.getTextContent());
									}
									else if(childElement.getTagName().compareToIgnoreCase("y1") == 0) {
										y1 = Integer.parseInt(childElement.getTextContent());
									}
									else if(childElement.getTagName().compareToIgnoreCase("x2") == 0) {
										x2 = Integer.parseInt(childElement.getTextContent());
									}
									else if(childElement.getTagName().compareToIgnoreCase("y2") == 0) {
										y2 = Integer.parseInt(childElement.getTextContent());
									}
									else if(childElement.getTagName().compareToIgnoreCase("feature") == 0) {
										// Read instance feature
										int featureDim = Integer.parseInt(childElement.getAttribute("dim"));
										if(dim >= 0 && featureDim != dim) {
											System.out.println("ERROR - read dimension feature " + featureDim + " vs " + dim);
											System.exit(0);
										}
										String stringFeature = childElement.getTextContent();
										feature = readFeature(stringFeature, featureDim);
									}
								}
							}

							Integer[] region = {x1, y1, x2, y2};

							if(index != -1) {
								addInstance(index, feature);
								addRegion(index, region);
							}
							else {
								addInstance(feature);
								addRegion(region);
							}
						}
					}				
				}

				if(numberOfInstances != -1 && numberOfInstances != numberOfInstances()) {
					System.out.println("Error in readXMLFile - number of instances incorrect " + numberOfInstances + " != " + numberOfInstances());
				}
			}
			catch (final ParserConfigurationException e) {
				e.printStackTrace();
			}
			catch (final SAXException e) {
				e.printStackTrace();
			}
			catch (final IOException e) {
				e.printStackTrace();
			}
		}
		else {
			System.out.println("\nError file " + file.getAbsolutePath() + " does not exist");
			System.exit(0);
		}
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @return the region
	 */
	public Integer[] getRegion(int index) {
		return regions.get(index);
	}

	/**
	 * @param regions the regions to set
	 */
	public void setRegions(int index, Integer[] region) {
		regions.set(index, region);
	}

	/**
	 * @return the regions
	 */
	public List<Integer[]> getRegions() {
		return regions;
	}

	/**
	 * @param regions the regions to set
	 */
	public void setRegions(List<Integer[]> regions) {
		this.regions = regions;
	}

}
