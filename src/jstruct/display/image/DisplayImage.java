/**
 * 
 */
package jstruct.display.image;

import java.awt.Image;

import javax.swing.JFrame;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class DisplayImage extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2563450532387569797L;
	
	private ImagePanel pan = null;
	private int imageDisplayType = 0;

	public DisplayImage(){
		this.setTitle("DisplayImage");
		this.setSize(400, 400);
		this.setLocationRelativeTo(null);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);             
		this.setVisible(true);
		this.setResizable(true);
		
		//Instanciation d'un objet JPanel
		pan = new ImagePanel();
		pan.setImageDisplayType(imageDisplayType);
 
	    //On prévient notre JFrame que notre JPanel sera son content pane
	    this.setContentPane(pan);               
	    this.setVisible(true);
	}

	/**
	 * 
	 * @param image
	 */
	public void display(Image image) {
		pan.setImage(image);
		pan.repaint();
	}
	
	/**
	 * 
	 * @param image
	 * @param region
	 */
	public void display(Image image, Integer[] region) {
		pan.setImage(image);
		pan.setRegion(region);
		pan.repaint();
	}

	/**
	 * @return the imageDisplayType
	 */
	public int getImageDisplayType() {
		return imageDisplayType;
	}

	/**
	 * @param imageDisplayType the imageDisplayType to set
	 */
	public void setImageDisplayType(int imageDisplayType) {
		this.imageDisplayType = imageDisplayType;
		pan.setImageDisplayType(imageDisplayType);
		pan.repaint();
	}
	
	public void close() {
		this.dispose();
	}
	
	
}
