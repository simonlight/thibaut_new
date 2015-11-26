package fr.durandt.jstruct.ssvm.ranking.nico;

import java.util.List;
import java.util.Set;

public interface StructuralType<X,Y> {
	
	public double[] psi(X x , Y y);
	public double delta(Y yi , Y y);
	public Set<Y> enumerateY();
	//public void printinfo(List<STrainingSample<X,Y>> list, int epoch);
	public String evaluation(List<Y> predictions , List<Y> gt);

}
