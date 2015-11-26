package fr.durandt.jstruct.ssvm.ranking.nico;


public class RankingData {

	public double[] vectors;
	public int ranking_id;
	
	public RankingData(double[] vectors, int ranking_id) {
		super();
		this.vectors = vectors;
		this.ranking_id = ranking_id;
	}
	
}
