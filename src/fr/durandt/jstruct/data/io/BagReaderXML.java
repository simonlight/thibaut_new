/**
 * 
 */
package fr.durandt.jstruct.data.io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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

import fr.durandt.jstruct.extern.pca.PrincipalComponentAnalysis;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageRegion;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * Class to read a list of bag training samples
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class BagReaderXML {

	public static void main(String[] args) {
		//File file = new File("/Volumes/Eclipse/LIP6/simulation/UIUCSports/Split_1/files_BagImageRegion/caffe_hybrid_caffe_7_relu/multiclass_train_scale_100.xml");
		//readBagImageRegion(file, 4096, true, true, null, true, 0);

		File file = new File("/Volumes/Eclipse/LIP6/simulation/SiftFlowDataset/files/weakly_segmentation/caffe_vgg19_6_relu_vd_monoscale_384/k_0.500000/weak_segmentation_test.xml");
		List<STrainingSample<BagImageSeg, Integer[]>> sample = readBagImageSegmentation(file, 4096, true, true, null, true, 0);

		for(int i=0; i<sample.size(); i++) {
			System.out.println(i + "\t" + Arrays.toString(sample.get(i).output));
			sample.get(i).input.print();
		}

	}

	public static List<STrainingSample<BagImageRegion, Integer>> readBagImageRegion(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {

		List<STrainingSample<BagImageRegion, Integer>> list = null;

		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

		try {
			DocumentBuilder builder = factory.newDocumentBuilder();
			Document document = builder.parse(file);
			Element racine = document.getDocumentElement();

			// Read the number of bags nbBags
			int nbBags = Integer.parseInt(racine.getAttribute("size"));

			// Initialize the list of bags
			list = new ArrayList<STrainingSample<BagImageRegion, Integer>>(nbBags);
			int nbInstancesAll = 0;

			NodeList racineNodes = racine.getChildNodes();
			int nbRacineNodes = racineNodes.getLength();

			for(int i=0; i<nbRacineNodes; i++) {
				if(racineNodes.item(i).getNodeType() == Node.ELEMENT_NODE) {
					Element e = (Element) racineNodes.item(i);
					if(e.getTagName().compareToIgnoreCase("example") == 0) {
						NodeList childNoeuds = e.getChildNodes();
						int nbChildNoeuds = childNoeuds.getLength();

						String featureFile = null;
						String name = null;
						int label = 0;

						for(int j=0; j<nbChildNoeuds; j++) {
							if(childNoeuds.item(j).getNodeType() == Node.ELEMENT_NODE) {
								Element childElement = (Element) childNoeuds.item(j);

								if(verbose > 2) System.out.println(j + "\t" + childElement);

								if(childElement.getTagName().compareToIgnoreCase("featurefile") == 0) {
									featureFile = childElement.getTextContent();
								}
								else if(childElement.getTagName().compareToIgnoreCase("imagename") == 0) {
									name = childElement.getTextContent();
								}
								else if(childElement.getTagName().compareToIgnoreCase("output") == 0) {
									label = Integer.parseInt(childElement.getTextContent());
								}
							}
						}

						BagImageRegion bag = new BagImageRegion();
						if(featureFile != null && withFeatures) {
							bag.readXMLFile(new File(featureFile), dim, 0);
						}
						if(name != null) {
							bag.setName(name);
						}

						// Add the new bag to the list
						list.add(new STrainingSample<BagImageRegion, Integer>(bag, label));
						nbInstancesAll += bag.numberOfInstances();

						System.out.print(".");
						if(list.size()>0 && list.size() % 100 == 0) System.out.print(list.size());
					}
				}
			}
			System.out.println("*\nnumber of bags= " + list.size() + "\tnumber of instances= " + nbInstancesAll + "\taverage number of instances per bag= " + ((double)nbInstancesAll/(double)list.size()));
		}
		catch (ParserConfigurationException e) {
			e.printStackTrace();
		}
		catch (SAXException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}

		// Pre-treatment of the instances
		for(STrainingSample<BagImageRegion, Integer> ts : list) {
			for(int i=0; i<ts.input.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.input.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.input.setInstance(i, pca.sampleToEigenSpace(ts.input.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.input.setInstance(i,VectorOp.addValeur(ts.input.getInstance(i),1));
				}
			}
		}

		return list;
	}

	public static List<STrainingSample<BagImageSeg, Integer[]>> readBagImageSegmentation(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {

		List<STrainingSample<BagImageSeg, Integer[]>> list = null;

		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

		try {
			DocumentBuilder builder = factory.newDocumentBuilder();
			Document document = builder.parse(file);
			Element racine = document.getDocumentElement();

			// Read the number of bags nbBags
			int nbBags = Integer.parseInt(racine.getAttribute("size"));

			// Initialize the list of bags
			list = new ArrayList<STrainingSample<BagImageSeg, Integer[]>>(nbBags);
			int nbInstancesAll = 0;

			NodeList racineNodes = racine.getChildNodes();
			int nbRacineNodes = racineNodes.getLength();

			for(int i=0; i<nbRacineNodes; i++) {
				if(racineNodes.item(i).getNodeType() == Node.ELEMENT_NODE) {
					Element e = (Element) racineNodes.item(i);
					if(e.getTagName().compareToIgnoreCase("example") == 0) {
						NodeList childNoeuds = e.getChildNodes();
						int nbChildNoeuds = childNoeuds.getLength();

						String featureFile = null;
						String name = null;
						Integer[] output = null;
						String gtmaskFile = null;
						int index = -1;

						for(int j=0; j<nbChildNoeuds; j++) {
							if(childNoeuds.item(j).getNodeType() == Node.ELEMENT_NODE) {
								Element childElement = (Element) childNoeuds.item(j);

								if(verbose > 2) System.out.println(j + "\t" + childElement);

								if(childElement.getTagName().compareToIgnoreCase("featurefile") == 0) {
									// Read the path of the feature file
									featureFile = childElement.getTextContent();
								}
								else if(childElement.getTagName().compareToIgnoreCase("imagename") == 0) {
									// Read the path of the image
									name = childElement.getTextContent();
								}
								else if(childElement.getTagName().compareToIgnoreCase("output") == 0) {
									// Read the output
									int outputDim = Integer.parseInt(childElement.getAttribute("dim"));
									String outputString = childElement.getTextContent();
									output = readWeaklySegmentationOutput(outputString, outputDim);
								}
								else if(childElement.getTagName().compareToIgnoreCase("gtmask") == 0) {
									// Read the ground truth mask file
									gtmaskFile = childElement.getTextContent();
								}
								else if(childElement.getTagName().compareToIgnoreCase("index") == 0) {
									// Read index of instance
									index = Integer.parseInt(childElement.getTextContent());
								}
							}
						}

						// Create a new bag and set attributes
						BagImageSeg bag = new BagImageSeg();
						if(featureFile != null && withFeatures) {
							bag.readXMLFile(new File(featureFile), dim, 0);
						}
						if(name != null) {
							bag.setName(name);
						}
						if(gtmaskFile != null) {
							bag.setGtMaskFile(gtmaskFile);
						}

						// Add the new bag to the list
						if(index >= 0) {
							list.add(index, new STrainingSample<BagImageSeg, Integer[]>(bag, output));
						}
						else {
							list.add(new STrainingSample<BagImageSeg, Integer[]>(bag, output));
						}

						// Count the number of instances
						nbInstancesAll += bag.numberOfInstances();

						System.out.print(".");
						if(list.size()>0 && list.size() % 100 == 0) System.out.print(list.size());
					}
				}
			}
			System.out.println("*\nnumber of bags= " + list.size() + "\tnumber of instances= " + nbInstancesAll + "\taverage number of instances per bag= " + ((double)nbInstancesAll/(double)list.size()));
		}
		catch (ParserConfigurationException e) {
			e.printStackTrace();
		}
		catch (SAXException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}

		// Pre-treatment of the instances
		for(STrainingSample<BagImageSeg, Integer[]> ts : list) {
			for(int i=0; i<ts.input.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.input.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.input.setInstance(i, pca.sampleToEigenSpace(ts.input.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.input.setInstance(i,VectorOp.addValeur(ts.input.getInstance(i),1));
				}
			}
		}

		return list;
	}

	/**
	 * Read the output in weakly-supervised segmentation case
	 * @param s the sting with the output
	 * @param dim dimension of the output (= number of classes)
	 * @return array of the output
	 */
	private static Integer[] readWeaklySegmentationOutput(String s, int dim) {
		Integer[] output = new Integer[dim];
		StringTokenizer st = new StringTokenizer(s);
		if(st.countTokens() != dim) {
			System.out.println("Error - number of tokens incorrect " + st.countTokens() + " vs " + dim*dim);
			System.exit(0);
		}
		for(int i=0; i<dim; i++) {
			output[i] = Integer.parseInt(st.nextToken());
		}
		return output;
	}

}
