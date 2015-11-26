package fr.durandt.jstruct.ssvm.ranking.nico;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;


// Ranking instantiation in the class mode 
// X: list of images  
// Y: A ranking 

public class RankingType implements StructuralType<List<double[]>, List<Integer>> {

	int dim;
	List<Integer> gtRanking;
	List<Integer> gtOrdering;
	List<Integer> gtLabels;
	public int nbPlus;
	int nbMinus;
	
	public RankingType(int dim, int nbPlus, int nbMinus, List<Integer> gtRanking) {
		super();
		this.dim = dim;
		this.nbPlus = nbPlus;
		this.nbMinus = nbMinus;
		this.gtRanking = gtRanking;
		this.gtOrdering = convertRankOrdering(gtRanking);
		gtLabels = getLabelsFromRanking(gtRanking,nbPlus);

		System.out.println("gtRanking \t" + gtRanking);
		System.out.println("gtLabels \t" + gtLabels);
		System.out.println("gtOrdering \t" + gtOrdering);
		System.out.println("LabelsFromOrdering \t" + getLabelsFromOrdering(gtOrdering));
		
	}
	
	public List<Integer> getLabelsFromOrdering(List<Integer> ordering){
		// Input : List with indices sorted (in decreasing order) containing the order of each example
		List<Integer> res = new ArrayList<Integer>();
		for(int i=0;i<ordering.size();i++){
			if(gtLabels.get(ordering.get(i))==1)
				res.add(1);
			else
				res.add(-1);
		}
		
		return res;
	}
	
	public List<Integer> getLabelsFromRanking(List<Integer> ranking , int nbPlus){
		// Input : List with indices containing the rank of each example 
		List<Integer> res = new ArrayList<Integer>();
		for(int i=0;i<ranking.size();i++){
			if(ranking.get(i)<nbPlus)
				res.add(1);
			else
				res.add(-1);
		}
		
		return res;
	}

	@Override
	public double[] psi(List<double[]> l,	List<Integer> y) {
		// Input Y : supposed to be list containing the ordered indices of examples (as stored in gtOrdering)
		// Y Conversion
		y = convertRankOrdering(y);
		// Y is now a list with the rank of each example (as stored in gtRanking)
		
		long start = System.currentTimeMillis();
		//System.out.println("calcul PSI  - taille l : "+l.size()+" y="+y+" GT labels="+gtLabels);
		
		double [] res = new double[dim];
		double cpt = 0;
		for(int i=0;i<l.size();i++){
			//int pi = y.get(i); // Position in the initial list of the ist example in y 
			int pi = i; // Position in the initial list of the ist example in y 
			int li = gtLabels.get(i); // Label of this example
			//System.out.println(" i="+i);
			if(li==1){
				for(int j=0;j<l.size();j++){
					int pj = j; // Position in the initial list of the jst example in y 
					int lj = gtLabels.get(j); // Label of this example	
					//System.out.println(" li="+li+" lj="+lj+" pi="+pi+" pj="+pj+" i="+i+" j="+j);
					//if(j!=i && !y.get(i).equals( y.get(j))){
					if(j!=i && lj==-1){	
						//System.out.println(" li="+li+" lj="+lj+" pi="+pi+" pj="+pj+" i="+i+" j="+j);
						int yij =1;
						if(y.get(j)<y.get(i))
							yij = -1;
						//System.out.println(" yij="+yij);
						double[] v1 = l.get(pi);
						double[] v2 = l.get(pj);
						for(int k=0;k<dim;k++){
							res[k] += yij * (v1[k] - v2[k]);
							//if(k==0)
							//	System.out.println("res["+k+"]="+res[k]+" v1[k]="+v1[k]+" v2[k]="+v2[k]);
						}
						

						cpt++;
					}

				}
			}
		}
		for(int k=0;k<dim;k++){
			res[k] /= cpt;
		}

		long et = System.currentTimeMillis()-start;
		//System.out.println(" FIN calcul PSI  - temps "+et+" nb pairs="+cpt);//+" res="+Arrays.toString(res));
		
		return res;
	}

	@Override
	public double delta(List<Integer> yi, List<Integer> y) {
		// Input Y : supposed to be list containing the ordered indices of examples (as stored in gtOrdering)
		// Yi label of all examples
		return 1.0-ap(yi, y   );
	}

	@Override
	public Set<List<Integer>> enumerateY() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String evaluation(List<List<Integer>> predictions,List<List<Integer>> gt) {
		// TODO Auto-generated method stub
		return "AP="+delta(gt.get(0),predictions.get(0));
	}

	public double ap(List<Integer> yi, List<Integer> y /*, int nbPlus*/){
		// Input Y : supposed to be list containing the ordered indices of examples (as stored in gtOrdering)
		//System.out.println(" CALCUL AP - NbPLus="+nbPlus);
		
		List<Double> precision = new ArrayList<Double>();
		List<Double> recall = new ArrayList<Double>();
		
//		List<Integer> labelsGT = new ArrayList<Integer>();
//		for(int i=0;i<y.size();i++){
//			labelsGT.add(gtLabels.get(y.get(i)));
//		}
//		System.out.println(" Labels for predicted : ");
//		System.out.println(labelsGT);
		
		List<Integer> labelsy = new ArrayList<Integer>(Collections.nCopies(y.size(), 0));
		//System.out.println(" y="+y);
		int nbPlus=0;
		for(int i=0;i<y.size();i++){
			//labelsy.set(y.get(i), gtLabels.get(i));
			labelsy.set(i, yi.get(y.get(i)));
			if(yi.get(y.get(i))==1)
					nbPlus++;
		}
		//System.out.println("nbPlus="+nbPlus);
		//System.out.println("evaluating:"+y+" labels y="+labelsy+" yi="+yi);
		
		int top=0;
		for(int i=0;i<y.size();i++){
			//if(gtLabels.get(y.get(i)) == 1 ){
			if(labelsy.get(i) == 1 ){
				top++;
			}
			precision.add(top/(double)(i+1));
			recall.add(top/(double)nbPlus);
		}
		//System.out.println("y="+y);
		//System.out.println(precision);
		//System.out.println(recall);
		
		
		double[][] RP = new double[2][precision.size()+1];
		RP[0][0] = 0.0;
		RP[1][0] = 1.0;
		for(int j = 1 ; j <= recall.size(); j++)
			RP[0][j] = recall.get(j-1);
		
		for(int j = 1 ; j <= recall.size(); j++)
			RP[1][j] = precision.get(j-1);
		
		// Calcul du MAP par la m�thode des trap�zes
		double AP = 0.0;
		
		for(int j = 0 ; j < RP[0].length-1; j++){
			AP += (RP[1][j+1]+RP[1][j]) * (RP[0][j+1]-RP[0][j]) /2.0;
			
		}
		
		//System.out.println("AP="+AP*100.0);
		return AP;
	}

	
	public static List<Integer> convertRankOrdering(List<Integer> list){
		// Input : ordered List (ex [2,0,1,3])
		// Output : rank of each element : 0=>1 1=>2 2=>0 3=>4 
		// This is symmetric, switiching input/output works
		List<Integer> res= new ArrayList<Integer>(Collections.nCopies(list.size(), 0));
		
		for(int i=0;i<res.size();i++){
			res.set(list.get(i), i);
		}


		return res;
	}
	
	
//	public List<Integer> convertRankOrdering(List<Integer> list){
//		// Input :  rank of each element : 0=>1 1=>2 2=>0 3=>4 
//		// Output : ordered List (ex [2,0,1,3])
//		List<Integer> res= new ArrayList<Integer>(Collections.nCopies(list.size(), 0));
//		
//		for(int i=0;i<res.size();i++){
//			res.set(list.get(i), i);
//		}
//
//
//		return res;
//	}



}
