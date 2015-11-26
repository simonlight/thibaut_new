package jstruct.data.voc2011;


public class VOC2011 {

	/**
	 * 
	 * @return list of the classes of PASCAL VOC 2011
	 */
	public static String[] getClasses() {
		String[] listCls = new String[20];
		listCls[0] = "aeroplane";
		listCls[1] = "bicycle";
		listCls[2] = "bird";
		listCls[3] = "boat";
		listCls[4] = "bottle";
		listCls[5] = "bus";
		listCls[6] = "car";
		listCls[7] = "cat";
		listCls[8] = "chair";
		listCls[9] = "cow";
		listCls[10] = "diningtable";
		listCls[11] = "dog";
		listCls[12] = "horse";
		listCls[13] = "motorbike";
		listCls[14] = "person";
		listCls[15] = "pottedplant";
		listCls[16] = "sheep";
		listCls[17] = "sofa";
		listCls[18] = "train";
		listCls[19] = "tvmonitor";
		return listCls;
	}
	
	public static String[] getActionClasses() {
		String[] listCls = new String[10];
		listCls[0] = "jumping";
		listCls[1] = "phoning";
		listCls[2] = "playinginstrument";
		listCls[3] = "reading";
		listCls[4] = "ridingbike";
		listCls[5] = "ridinghorse";
		listCls[6] = "running";
		listCls[7] = "takingphoto";
		listCls[8] = "usingcomputer";
		listCls[9] = "walking";
		return listCls;
	}
}
