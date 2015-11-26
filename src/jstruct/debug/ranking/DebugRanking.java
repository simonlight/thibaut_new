/**
 * 
 */
package jstruct.debug.ranking;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.ssvm.ranking.RankingOutput;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class DebugRanking {

	public static void main(String[] args) {

		RankingOutput ro = new RankingOutput();

		List<Integer> labelsGT = new ArrayList<Integer>();
		labelsGT.add(1);
		labelsGT.add(-1);
		labelsGT.add(1);
		labelsGT.add(1);
		labelsGT.add(-1);

		ro.initialize(labelsGT);
		ro.printInfo();

		Integer[] predict = {2,1,3,4,5};

		RankingOutput rop = new RankingOutput(predict, ro.getnPos(), ro.getnNeg());
		rop.printInfo();
		
		System.out.println("ap= " + RankingOutput.averagePrecision(ro, rop));

	}

}
