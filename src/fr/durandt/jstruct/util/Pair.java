/**
 * 
 */
package fr.durandt.jstruct.util;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class Pair<K,V extends Comparable<V>> implements Comparable<Pair<K,V>> {
	
	private K key;
	private V value;
	
	public K getKey() {
		return key;
	}
	
	public V getValue() {
		return value;
	}

	public Pair(K key, V value) {
		super();
		this.key = key;
		this.value = value;
	}

	@Override
	public int compareTo(Pair<K, V> o) {
		return value.compareTo(o.value);
	}

	/**
	 * @param value the value to set
	 */
	public void setValue(V value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return "(" + key + ", " + value + ")";
	}
	

}