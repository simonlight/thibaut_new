package fr.durandt.jstruct.struct;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import fr.lip6.jkernelmachines.type.TrainingSample;

public class STrainingSample<X,Y> implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -3221897092308838680L;
	
	public X input;
	public Y output; 
	
	public STrainingSample(X input, Y output) {
		this.input = input;
		this.output = output;
	}
	
	/**
	 * Conversion of TrainingSample &lt double[] &gt to STrainingSample &lt double[], Integer &gt
	 * @param l
	 * @return
	 */
	public static List<STrainingSample<double[], Integer>> trainingSample2STrainingSample(List<TrainingSample<double[]>> l) {
		List<STrainingSample<double[], Integer>> newList = new ArrayList<STrainingSample<double[], Integer>>(l.size()); 
		for(int i=0; i<l.size(); i++) {
			newList.add(new STrainingSample<double[], Integer>(l.get(i).sample, l.get(i).label));
		}
		return newList;
	}
}
