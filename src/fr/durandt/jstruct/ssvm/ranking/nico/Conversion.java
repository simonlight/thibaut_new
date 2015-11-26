package fr.durandt.jstruct.ssvm.ranking.nico;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;

public class Conversion <X,Y>{

	public List<STrainingSample<double[],String>> convert(List<STrainingSample<X,Y>> lts){
		List<STrainingSample<double[],String>> res = new ArrayList<STrainingSample<double[],String>>();
		for(STrainingSample<X,Y> ts : lts){
			res.add(new STrainingSample<double[], String>((double[])ts.input, (String)ts.output));
		}
		
		return res;
	}
	
	public static List<Integer> fusionList (List<Integer> l1, List<Integer> l2, List<Integer> pos){	
		if(l2.size() != pos.size()){
			System.err.println(" Error fusionList ! l2 must be the same size than pos !");
			return null;
		}
		
		//int taille = l1.size() + l2.size();
		//List<Integer> res = new ArrayList<Integer>(Collections.nCopies(taille, 0));
		
		List<Integer> res = new ArrayList<Integer>(l1);
		//Collections.copy(res, l1);
		
//		System.out.println(l1);
//		System.out.println(l2);
//		System.out.println(pos);
		
		//int min=Integer.MAX_VALUE;
		
		for(int i=0;i<l2.size();i++){
			int dec=0;
			for(int j=0;j<i;j++){
				if(pos.get(j)<pos.get(i))
					dec++;
			}
			//System.out.println(res+" ajout "+l2.get(i)+" a pos "+(pos.get(i)+dec));
			res.add(pos.get(i)+dec,l2.get(i));
		}
		
		//System.out.println(res);
		return res;
	}
}
