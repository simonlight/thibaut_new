package fr.durandt.jstruct.variable;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.StringTokenizer;

import javax.imageio.ImageIO;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import jstruct.display.image.ImageRGBOp;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;



/**
 * Bag for weakly-supervised segmentation
 * 
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class BagImageSeg extends BagImageRegion {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	// File of the ground truth mask
	protected String gtMaskFile;
	// File with the superpixels
	protected String superpixelsFile;

	// File with the neigbhors / structure of the graph
	protected String neigbhorsFile;

	// Matrix n*n where n is the number of superpixels
	// 1 --> pixels are neigbhors
	// 0 --> pixels are not neigbhors
	protected Integer[][] neigbhors;

	// Image of the ground truth mask
	protected BufferedImage gtMask;

	// Image of the superpixels
	protected BufferedImage superpixels;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	public BagImageSeg() {
		super();
		// ground truth mask
		gtMaskFile = null;
		gtMask = null;
		// superpixels
		superpixelsFile = null;
		superpixels = null;
		// neigbhors
		neigbhors = null;

	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @return the maskGTFile
	 */
	public String getGtMaskFile() {
		return gtMaskFile;
	}

	/**
	 * @param maskGTFile the maskGTFile to set
	 */
	public void setGtMaskFile(String gtMaskFile) {
		this.gtMaskFile = gtMaskFile;
	}

	/**
	 * @return the superpixelsFile
	 */
	public String getSuperpixelsFile() {
		return superpixelsFile;
	}

	/**
	 * @param superpixelsFile the superpixelsFile to set
	 */
	public void setSuperpixelsFile(String superpixelsFile) {
		this.superpixelsFile = superpixelsFile;
	}

	/**
	 * @return the neigbhors
	 */
	public Integer[][] getNeigbhors() {
		return neigbhors;
	}

	/**
	 * @param neigbhors the neigbhors to set
	 */
	public void setNeigbhors(Integer[][] neigbhors) {
		this.neigbhors = neigbhors;
	}

	/**
	 * @return the neigbhorsFile
	 */
	public String getNeigbhorsFile() {
		return neigbhorsFile;
	}

	/**
	 * @param neigbhorsFile the neigbhorsFile to set
	 */
	public void setNeigbhorsFile(String neigbhorsFile) {
		this.neigbhorsFile = neigbhorsFile;
	}

	public String toString() {
		String s = super.toString();
		if(gtMaskFile != null) {
			s += "\ngtMaskFile: " + gtMaskFile;
		}
		if(superpixelsFile != null) {
			s += "\tsuperpixelsFile: " + superpixelsFile;
		}
		if(neigbhorsFile != null) {
			s += "\tneigbhorsFile: " + neigbhorsFile;
		}
		return s;
	}

	public void printNeigbhors() {
		System.out.println("neigbhors");
		if(neigbhors != null) {
			for(int i=0; i<neigbhors.length; i++) {
				for(int j=0; j<neigbhors[i].length; j++) {
					System.out.print(neigbhors[i][j] + "\t");
				}
				System.out.println();
			}
		}
		else {
			System.out.println("neigbhors is null");
		}
	}

	protected BufferedImage readImage(String file) {
		BufferedImage image = null;
		try {
			// Read the image
			image = ImageIO.read(new File(file));
		} catch (IOException e) {
			e.printStackTrace();
		}
		// Coversion in RGB format
		BufferedImage img = ImageRGBOp.copyImage(image,BufferedImage.TYPE_INT_RGB);
		return img;
	}

	public void readSuperpixelsImage() {
		superpixels = readImage(superpixelsFile);
	}

	public void readGtMask() {
		gtMask = readImage(gtMaskFile);
	}

	/**
	 * @return the maskGT
	 */
	public BufferedImage getGtMask() {
		if(gtMask == null) {
			readGtMask();
		}
		return gtMask;
	}

	/**
	 * @param maskGT the maskGT to set
	 */
	public void setMaskGT(BufferedImage gtMask) {
		this.gtMask = gtMask;
	}

	/**
	 * @return the superpixels
	 */
	public BufferedImage getSuperpixels() {
		return superpixels;
	}

	/**
	 * @param superpixels the superpixels to set
	 */
	public void setSuperpixels(BufferedImage superpixels) {
		this.superpixels = superpixels;
	}

	public BufferedImage predictedMask(Integer[] predictLabels) {
		if(superpixels == null) {
			readSuperpixelsImage();
		}
		BufferedImage img = ImageRGBOp.copyImage(superpixels, BufferedImage.TYPE_INT_RGB);
		for(int i=0; i<img.getHeight(); i++) {
			for(int j=0; j<img.getWidth(); j++) {
				int pixel = img.getRGB(j, i);
				int blue = (pixel) & 0xff;
				ImageRGBOp.setPixelRGB(img, j, i, predictLabels[blue-1]+1, predictLabels[blue-1]+1, predictLabels[blue-1]+1);
			}
		}
		return img;
	}

	public double pixelAccuracy(Integer[] predictLabels) {
		BufferedImage predict = predictedMask(predictLabels);
		int good = 0;
		for(int i=0; i<predict.getHeight(); i++) {
			for(int j=0; j<predict.getWidth(); j++) {
				int pixel = predict.getRGB(j, i);
				int predictLabel = (pixel) & 0xff;
				pixel = gtMask.getRGB(j, i);
				int gtLabel = (pixel) & 0xff;
				if(gtLabel == predictLabel) {
					good++;
				}
			}
		}
		double accuracy = (double)good / (double)(predict.getHeight() * predict.getWidth());
		System.out.println("accuracy= " + accuracy + "\t(" + good + "/" + (predict.getHeight() * predict.getWidth()) + ")");
		return accuracy;
	}

	public void readXMLFile(File file, int dim, int verbose) {

		if(file.exists()) {
			// Etape 1 : récupération d'une instance de la classe "DocumentBuilderFactory"
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

			try {
				// Etape 2 : création d'un parseur
				DocumentBuilder builder = factory.newDocumentBuilder();

				//Etape 3 : création d'un Document
				Document document = builder.parse(file);

				// Etape 4 : récupération de l'Element racine
				Element racine = document.getDocumentElement();

				// Etape 5 : récupération des personnes
				NodeList racineNoeuds = racine.getChildNodes();
				int nbRacineNoeuds = racineNoeuds.getLength();

				int numberOfInstances = -1;

				for(int i=0; i<nbRacineNoeuds; i++) {
					if(racineNoeuds.item(i).getNodeType() == Node.ELEMENT_NODE) {
						Element e = (Element) racineNoeuds.item(i);
						if(verbose > 2) System.out.println(i + "\t" + e + "\t" + e.getTextContent());

						if(e.getTagName().compareToIgnoreCase("name") == 0) {
							String name = e.getTextContent();
							setName(name);
						}
						else if(e.getTagName().compareToIgnoreCase("height") == 0) {
							int height = Integer.parseInt(e.getTextContent());
							setHeight(height);
						}
						else if(e.getTagName().compareToIgnoreCase("width") == 0) {
							int width = Integer.parseInt(e.getTextContent());
							setWidth(width);
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
							imageFile = e.getTextContent();
						}
						else if(e.getTagName().compareToIgnoreCase("superpixelfile") == 0) {
							superpixelsFile = e.getTextContent();
						}
						else if(e.getTagName().compareToIgnoreCase("instance") == 0) {
							NodeList childNoeuds = e.getChildNodes();
							int nbChildNoeuds = childNoeuds.getLength();

							int x1 = -1;
							int y1 = -1;
							int x2 = -1;
							int y2 = -1;

							// Index of super-pixel
							int index = -1;

							// Representation of super-pixel
							double[] feature = null;

							for(int j=0; j<nbChildNoeuds; j++) {
								if(childNoeuds.item(j).getNodeType() == Node.ELEMENT_NODE) {
									Element childElement = (Element) childNoeuds.item(j);

									if(verbose > 2) System.out.println(j + "\t" + childElement);

									if(childElement.getTagName().compareToIgnoreCase("x1") == 0) {
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
									else if(childElement.getTagName().compareToIgnoreCase("index") == 0) {
										index = Integer.parseInt(childElement.getTextContent());
									}
									else if(childElement.getTagName().compareToIgnoreCase("feature") == 0) {
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
							addRegion(index,region);

							addInstance(index,feature);
						}
						else if(e.getTagName().compareToIgnoreCase("graph") == 0) {
							int dimGraph = Integer.parseInt(e.getAttribute("dim"));
							String s = e.getTextContent();
							neigbhors = readGraph(s, dimGraph);
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

	protected Integer[][] readGraph(String s, int dim) {
		Integer[][] neigbhors = new Integer[dim][dim];
		StringTokenizer st = new StringTokenizer(s);
		if(st.countTokens() != dim*dim) {
			System.out.println("Error - number of tokens incorrect " + st.countTokens() + " vs " + dim*dim);
			System.exit(0);
		}
		for(int i=0; i<dim; i++) {
			for(int j=0; j<dim; j++) {
				neigbhors[i][j] = Integer.parseInt(st.nextToken());
			}
		}
		return neigbhors;
	}

	public void print() {
		super.print();
		if(superpixelsFile != null) {
			System.out.println("superpixels file: " + superpixelsFile);
		}
		printNeigbhors();
	}

}
