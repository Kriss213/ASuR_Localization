import numpy as np


class HoughMap:
    @staticmethod
    def hough_transform(binary_map:np.ndarray, angle_res:int):
        """
        Perform hough transformation on binary map.
        
        :param binary_map: The map to perform Hough transform on.
        :param angle_res: Angle resolution (in degrees) to create a vecotor in range [0; pi).
        
        :return angles, ht: angles and Hough transform matrix.
        """
        rows, cols = binary_map.shape
        angles_deg = np.arange(0, 180, angle_res)
        
        # calculate maximal rho value
        max_line_length = int(np.ceil( (rows**2 + cols**2)**(1/2) ))
        
        # construct accumulator array
        accumulator = np.zeros((max_line_length, len(angles_deg)))
        
        # perform hough transform for all map points
        for x in range(cols):
            for y in range(rows):
                if binary_map[y,x] != 1:
                    continue
                # parse all theta values and calculate rho
                for theta_index, theta_deg in enumerate(angles_deg):
                    theta = np.deg2rad(theta_deg)
                    rho = round(x * np.cos(theta) + y * np.sin(theta))
                    
                    # in accumulator array increase value for respective rho,angle cell
                    accumulator[rho, theta_index] += 1
                    
        return accumulator, angles_deg
    
    @staticmethod
    def hough_spectrum(ht:np.ndarray):
        """
        Get spectrum for Hough transform indicating line direction observed the most.
        
        :param ht: Hough transform.
        """
        
        spectrum = []
        rhos_len, angles_len = ht.shape
        for k in range(angles_len):
            val = 0
            for i in range(rhos_len):
                val += ht[i,k]**2 #
            spectrum.append(val)

        return np.array(spectrum)

    @staticmethod
    def circular_correlation(spectrum1:np.ndarray, spectrum2:np.ndarray):
        """
        Calculate correlation between 2 Hough spectrums.
        
        :param spectrum1: First Hough spectrum.
        :param spectrum2: Second Hough spectrum.
        
        """

        assert spectrum1.shape == spectrum2.shape, "Spectrums must be of the same size"
        
        theta_s = len(spectrum1)
        CC = np.zeros(theta_s)
        
        for k in range(theta_s):
            for i in range(theta_s):
                CC[k] += spectrum1[i] * spectrum2[(i+k) % theta_s]
        
        return CC
        
    @staticmethod
    def xy_spectrum(binary_map:np.ndarray):
        """
        Get X and Y spectrums from a map.
        
        :param binary_map: A binary map.
        """
        
        x_spectrum = np.sum(binary_map, axis=0)
        y_spectrum = np.sum(binary_map, axis=1)
        return x_spectrum, y_spectrum
    
    @staticmethod
    def translation_correlation(s1:np.ndarray, s2:np.ndarray):
       
        assert s1.shape == s2.shape, "Spectrums must be of the same size"
    
        translations = np.arange(-len(s1) +1, len(s1))
        correlation = np.zeros_like(translations)#np.zeros(2 * len(s1) - 1)
        
        for tau in translations:
            for k in range(len(s1)):
                if 0 <= k + tau < len(s1):
                    correlation[tau + len(s1) - 1] += s1[k + tau] * s2[k]
        
        return translations, correlation
        