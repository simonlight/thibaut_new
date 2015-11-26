/**
 * 
 */
package jstruct.display.image;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ImageRGBOp {

	public static BufferedImage normGray(BufferedImage src) {
		int max = 0;
		for(int i=0; i<src.getHeight(); i++) {
			for(int j=0; j<src.getWidth(); j++) {
				int pixel = src.getRGB(j, i);
				int blue = (pixel) & 0xff;
				if(blue > max) {
					max = blue;
				}
			}
		}
		double factor = 255/max;
		System.out.println("max= " + max + "\t factor= " + factor);
		BufferedImage output = copyImage(src, BufferedImage.TYPE_INT_RGB);
		for(int i=0; i<output.getHeight(); i++) {
			for(int j=0; j<output.getWidth(); j++) {
				//System.out.print("x= " + i + "\ty= " + j + "\t");
				int pixel = output.getRGB(j, i);
				//int red = (pixel >> 16) & 0xff;
				//int green = (pixel >> 8) & 0xff;
				int blue = (pixel) & 0xff;
				setPixelRGB(output, j, i, (int)(blue*factor), (int)(blue*factor), (int)(blue*factor));
			}
		}
		return output;
	}

	public static BufferedImage normGray(BufferedImage src, int max) {
		double factor = 255/max;
		System.out.println("max= " + max + "\t factor= " + factor);
		BufferedImage output = copyImage(src, BufferedImage.TYPE_INT_RGB);
		for(int i=0; i<output.getHeight(); i++) {
			for(int j=0; j<output.getWidth(); j++) {
				//System.out.print("x= " + i + "\ty= " + j + "\t");
				int pixel = output.getRGB(j, i);
				//int red = (pixel >> 16) & 0xff;
				//int green = (pixel >> 8) & 0xff;
				int blue = (pixel) & 0xff;
				setPixelRGB(output, j, i, (int)(blue*factor), (int)(blue*factor), (int)(blue*factor));
			}
		}
		return output;
	}

	public static BufferedImage normColor(BufferedImage src) {
		int max = 0;
		for(int i=0; i<src.getHeight(); i++) {
			for(int j=0; j<src.getWidth(); j++) {
				int pixel = src.getRGB(j, i);
				int blue = (pixel) & 0xff;
				if(blue > max) {
					max = blue;
				}
			}
		}
		double factor = 255/max;
		int n = (int)(Math.log(max)/Math.log(3)) + 1;
		System.out.println("max= " + max + "\t factor= " + factor + "\tn= " + n);
		Integer[][] tabConversion = getConversionTable(max,n,255);
		BufferedImage output = copyImage(src, BufferedImage.TYPE_INT_RGB);
		for(int i=0; i<output.getHeight(); i++) {
			for(int j=0; j<output.getWidth(); j++) {
				//System.out.print("x= " + i + "\ty= " + j + "\t");
				int pixel = output.getRGB(j, i);
				int blue = (pixel) & 0xff;
				setPixelRGB(output, j, i, tabConversion[blue-1][0], tabConversion[blue-1][1], tabConversion[blue-1][2]);
			}
		}
		return output;
	}

	public static void normColor(BufferedImage src, int max) {
		double factor = 255/max;
		int n = (int)(Math.log(max)/Math.log(3)) + 1;
		System.out.println("max= " + max + "\t factor= " + factor + "\tn= " + n);
		Integer[][] tabConversion = getConversionTable(max,n,255);
		for(int i=0; i<src.getHeight(); i++) {
			for(int j=0; j<src.getWidth(); j++) {
				//System.out.print("x= " + i + "\ty= " + j + "\t");
				int pixel = src.getRGB(j, i);
				int blue = (pixel) & 0xff;
				setPixelRGB(src, j, i, tabConversion[blue-1][0], tabConversion[blue-1][1], tabConversion[blue-1][2]);
			}
		}
	}

	private static Integer[] getValues(int max, int n) {
		Integer[] values = new Integer[n];
		for(int i=0; i<n; i++) {
			values[i] = i*max/(n-1);
		}
		return values;
	}

	private static Integer[][] getConversionTable(int max, int n, int valMax) {
		Integer[] values = getValues(valMax, n);
		System.out.println(Arrays.toString(values));
		Integer[][] tabConversion = new Integer[max][3];
		int i1=0, i2=0, i3=0; 
		for(int i=0; i<max; i++) {
			tabConversion[i][0] = values[i1];
			tabConversion[i][1] = values[i2];
			tabConversion[i][2] = values[i3];
			i1++;
			if(i1 == n) {
				i1 = 0;
				i2++;
			}
			if(i2 == n) {
				i2 = 0;
				i3++;
			}
			System.out.println(i + "\t" + Arrays.toString(tabConversion[i]));
		}
		return tabConversion;
	}

	/**
	 *  red component 0...255
	 *  green component 0...255
	 *  blue component 0...255
	 */
	public static void setPixelRGB(BufferedImage img, int x, int y, int r, int g, int b) {
		int col = (r << 16) | (g << 8) | b;
		img.setRGB(x, y, col);
	}

	public static BufferedImage copyImage(BufferedImage src, int imageType) {
		BufferedImage bufferedImage=new BufferedImage(src.getWidth(),src.getHeight(), imageType);
		Graphics2D g2 = bufferedImage.createGraphics();
		g2.drawImage(src, null, null);
		g2.dispose();
		return bufferedImage;
	}

}
