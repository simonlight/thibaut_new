/**
 * 
 */
package jstruct.display.image;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;

import javax.swing.JPanel;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ImagePanel extends JPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7765958654351696542L;

	private int imageDisplayType = 0;
	private Image image = null;
	private Integer[] region = null;

	@Override
	public void paintComponent(Graphics g){

		/*try {
			image = ImageIO.read(new File("/Users/thibautdurand/Pictures/lena.jpg"));
		} catch (IOException e) {
			e.printStackTrace();
		} */

		Graphics2D g2d = (Graphics2D) g;

		// Display the image
		if(image != null) {
			int x = 0;
			int y = 0;
			int width = 0;
			int height = 0;

			if(imageDisplayType == 0) {
				width = image.getWidth(this);
				height = image.getHeight(this);

				// Center the image in the JFrame
				x=((this.getWidth()-width)/2);
				y=((this.getHeight()-height)/2);
			}
			else if(imageDisplayType == 1) {
				// etire l'image
				width = this.getWidth();
				height = this.getHeight();
				x = 0;
				y = 0;
			}
			else if(imageDisplayType == 2) {
				// Get the dimension of the image
				width = image.getWidth(this);
				height = image.getHeight(this);

				// Compute the ratio between the image and the JFrame
				double ratioWidth = (double)width/(double)this.getWidth();
				double ratioHeight = (double)height/(double)this.getHeight();
				
				// Resize the dimension of the image
				if(ratioWidth > ratioHeight) {
					width = (int)(width / ratioWidth);
					height = (int)(height / ratioWidth);
				}
				else {
					width = (int)(width / ratioHeight);
					height = (int)(height / ratioHeight);
				}

				// Center the image in the JFrame
				x=((this.getWidth()-width)/2);
				y=((this.getHeight()-height)/2);
			}

			// Draw the image with the given x,y,width and height
			g2d.drawImage(image, x, y, width, height, this);
			
			if(region != null) {
				// Coordinates of the rectangle
				int x1 = 0;
				int y1 = 0;
				int rw = 0;
				int rh = 0;
				
				if(imageDisplayType == 0) {
					x1 = region[1]+x-1;
					y1 = region[0]+y-1;
					rw = region[3]-region[1];
					rh = region[2]-region[0];
				}
				else if(imageDisplayType == 1) {
					x1 = x;
					y1 = y;
					rw = (int)((region[3]-region[1])/((double)image.getWidth(this)/(double)width));
					rh = (int)((region[2]-region[0])/((double)image.getHeight(this)/(double)height));
				}
				else if(imageDisplayType == 2) {
					
					// Compute the ratio between the image and the JFrame
					double ratioWidth = (double)image.getWidth(this)/(double)this.getWidth();
					double ratioHeight = (double)image.getHeight(this)/(double)this.getHeight();
					double ratio = 0;
					// Resize the dimension of the region
					if(ratioWidth > ratioHeight) {
						ratio = ratioWidth;
					}
					else {
						ratio = ratioHeight;
					}
					x1 = (int)(region[1]/ratio) +x-1;
					y1 = (int)(region[0]/ratio) +y-1;
					rw = (int)((region[3]-region[1]) / ratio);
					rh = (int)((region[2]-region[0]) / ratio);
				}
				
				g2d.setColor(Color.RED);
				g2d.setStroke(new BasicStroke(5));
				//x1, y1, width, height
				g2d.drawRect(x1, y1, rw, rh);
			}
		}
	}

	/**
	 * @return the imageDisplayType
	 */
	public int getImageDisplayType() {
		return imageDisplayType;
	}

	/**
	 * Position de l'image sur le panel <br/>
	 * 0 - centrer avec dimension originale <br/>
	 * 1 - étirer (déformation de l'image) <br/>
	 * 2 - centrer et adapter dimension <br/>
	 * @param imageDisplayType 
	 */
	public void setImageDisplayType(int imageDisplayType) {
		this.imageDisplayType = imageDisplayType;
	}

	/**
	 * @return the image
	 */
	public Image getImage() {
		return image;
	}

	/**
	 * @param image the image to set
	 */
	public void setImage(Image image) {
		this.image = image;
	}

	/**
	 * @return the region
	 */
	public Integer[] getRegion() {
		return region;
	}

	/**
	 * @param region the region to set
	 */
	public void setRegion(Integer[] region) {
		this.region = region;
	}



}
