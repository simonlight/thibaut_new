package fr.durandt.jstruct.struct;

import java.io.Serializable;
import java.util.List;

public interface StructuralClassifier<X,Y> extends Serializable {
	public Y prediction(X x);
	public void train(List<STrainingSample<X,Y>> l);
}
