package Symil_thibaut_raw;
/**
 * 
 */


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SyMILGradientDescentBag extends SyMILGradientDescent<Bag,Integer> {

	protected int init = 0;

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<LatentRepresentation<Bag, Integer>> copy()
			throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double[] psi(Bag x, Integer h) {
		return x.getInstance(h);
	}

	@Override
	protected void init(List<TrainingSample<LatentRepresentation<Bag, Integer[]>>> l) {
		dim = l.get(0).sample.x.getInstance(0).length;
		if(init == 0) {
			for(TrainingSample<LatentRepresentation<Bag, Integer[]>> ts : l) {
				Integer[] h = new Integer[2];
				h[0] = 0;
				h[1] = ts.sample.x.getInstances().size()-1;
				ts.sample.h = h;
			}
		}
		else if(init == 1) {
			for(TrainingSample<LatentRepresentation<Bag, Integer[]>> ts : l) {
				Integer[] h = new Integer[2];
				h[0] = (int)(Math.random() * ts.sample.x.getInstances().size());
				h[1] = (int)(Math.random() * ts.sample.x.getInstances().size());
				ts.sample.h = h;
			}
		}
	}

	@Override
	protected Integer[] optimizeH(Bag x) {
		int hp = -1;
		int hm = -1;
		double maxVal = -Double.MAX_VALUE;
		double minVal = Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
			if(val < minVal) {
				minVal = val;
				hm = i;
			}
		}
		Integer[] hpredict = {hp,hm};
		return hpredict;
	}

	/**
	 * @return the init
	 */
	public int getInit() {
		return init;
	}

	/**
	 * @param init the init to set
	 */
	public void setInit(int init) {
		this.init = init;
	}

	public void writeResults(File file, List<TrainingSample<LatentRepresentation<Bag, Integer>>> l) {

		System.out.println("write results " + file.getAbsolutePath());
		file.getParentFile().mkdirs();

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			//bw.write("name \t gt \t ypredict \t hpredict \t score \n");
			for(int i=0; i<l.size(); i++){
				bw.write(l.get(i).sample.x.getName() + "\t" + l.get(i).label + "\t");
				double score = valueOf(l.get(i).sample);
				int ypredict = (int) Math.signum(score);
				Integer[] tmp = optimizeH(l.get(i).sample.x); 
				int hpredict = (ypredict > 0 ? tmp[0] : tmp[1]);	
				bw.write(ypredict + "\t" + hpredict + "\t" + score + "\n");
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}

}
