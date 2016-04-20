package fr.durandt.jstruct.variable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Create a class to represent a bag. 
 * The bag is composed of a set of instances
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class Bag {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Instances of the bag
	 */
	protected List<double[]> instances = null;

	/**
	 * Name of the bag
	 */
	protected String name;

	/**
	 * Global representation of the bag
	 */
	protected double[] bagFeature = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Constructor
	 */
	public Bag() {
		instances = new ArrayList<double[]>();
		name = "";
	}

	/**
	 * Copy constructor
	 * @param bag bag to copy
	 */
	public Bag(Bag bag) {
		instances = new ArrayList<double[]>();
		for(double[] instance : bag.getInstances()) {
			instances.add(instance.clone());
		}
		name = new String(bag.name);
		bagFeature = bag.bagFeature.clone();
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Add new instance to the bag
	 * @param instance
	 */
	public void addInstance(double[] instance) {
		instances.add(instance.clone());
	}

	/**
	 * Add new instance to the bag at a given index
	 * @param index in the list
	 * @param instance vector representation
	 */
	public void addInstance(int index, double[] instance) {
		instances.add(index, instance.clone());
	}

	/**
	 * Compute the mean of the instance features
	 * @return
	 */
	public double[] mean() {
		double[] mean = null;
		if(instances != null) {
			mean = new double[instances.get(0).length];
			for(double[] instance : instances) {
				for(int d=0; d<mean.length; d++) {
					mean[d] += instance[d];
				}
			}

			double n = instances.size();
			for(int d=0; d<mean.length; d++) {
				mean[d] /= n;
			}
		}
		return mean;
	}

	/**
	 * remove the index-th instance
	 * @param index
	 */
	public void removeInstance(int index) {
		if(index >= 0) {
			instances.remove(index);
		}
		else {
			System.out.println("index must be >= 0");
		}
	}

	/**
	 * Convert the feature string in array. Verify if the dimension is correct.
	 * @param s string of the feature
	 * @param dim dimension of the feature
	 * @return
	 */
	protected double[] readFeature(String s, int dim) {
		double[] feature = new double[dim];
		StringTokenizer st = new StringTokenizer(s);
		if(st.countTokens() != dim) {
			System.out.println("Error in readFeature - number of tokens incorrect " + st.countTokens() + " vs " + dim);
			System.exit(0);
		}
		// Convert the string in array of double
		for(int i=0; i<dim; i++) {
			feature[i] = Double.parseDouble(st.nextToken());
		}
		return feature;
	}

	/**
	 * Read the XML file of a bag
	 * @param file
	 */
	public void readXMLFile(File file) {
		readXMLFile(file, -1, 0);
	}
	
	/**
	 * Read the XML file of a bag
	 * @param file
	 * @param dim desired dimension feature (if dim is negative, there is no verification of the feature dimension)
	 * @param verbose
	 */
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
						else if(e.getTagName().compareToIgnoreCase("instance") == 0) {
							// Read an instance

							NodeList childNoeuds = e.getChildNodes();
							int nbChildNoeuds = childNoeuds.getLength();

							// Index of instance
							int index = -1;

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

							if(index != -1) {
								addInstance(index, feature);
							}
							else {
								addInstance(feature);
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

	/**
	 * @return the number of instances in the bag
	 */
	public int numberOfInstances() {
		if(instances != null) {
			return instances.size();
		}
		else {
			return 0;
		}
	}

	/**
	 * Print bag informations
	 */
	public void print() {
		// Print bag name and the number of instances
		System.out.println("Bag - name: " + name + "\tnumber of instances: " + numberOfInstances());
		// If the global bag feature exists, print the dimension of the feature
		if(bagFeature != null) {
			System.out.println("global bag feature dimension= " + bagFeature.length + "\t");
		}
		// If the instance features exist, print the dimension of each feature
		if(instances != null) {
			for(int i=0; i<numberOfInstances(); i++) {
				System.out.println(i + " - feature dimension= " + instances.get(i).length);
			}
		}
	}

	@Override
	public String toString() {
		String s = "name: " + name + "\tnumber of instances: " + numberOfInstances();
		return s;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Return the instance at given index 
	 * @param index
	 * @return instance at given index
	 */
	public double[] getInstance(int index) {
		return instances.get(index);
	}

	/**
	 * Set instance feature at given index
	 * @param index index to set
	 * @param feature new feature to set
	 */
	public void setInstance(int index, double[] feature) {
		instances.set(index,feature);
	}

	/**
	 * Set the list of instance features
	 * @param features list of instance features
	 */
	public void setInstances(List<double[]> features) {
		this.instances = features;
	}

	/**
	 * @return the list of instance features
	 */
	public List<double[]> getInstances() {
		return instances;
	}

	/**
	 * @return the name of the bag
	 */
	public String getName() {
		return name;
	}

	/**
	 * Set the name of the bag
	 * @param name of the bag
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return the bagFeature
	 */
	public double[] getBagFeature() {
		return bagFeature;
	}

	/**
	 * @param bagFeature the bagFeature to set
	 */
	public void setBagFeature(double[] bagFeature) {
		this.bagFeature = bagFeature;
	}

}
