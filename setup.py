from distutils.core import setup
setup(
  name = 'ObjectDetectionElsys',
  packages = ['ObjectDetectionElsys'],
  version = '0.1.1',
  license='MIT',
  description = 'Object detection and classification based on the YOLO & YOLOv2 algorithms implemented using TensorFlow',
  author = 'Evgeni Dimov',
  author_email = 'evgenidimovtues@gmail.com',
  url = 'https://github.com/GenchoBG/ObjectDetection',
  download_url = 'https://github.com/GenchoBG/ObjectDetection/archive/v_011.tar.gz',
  keywords = ['Computer Vision', 'Object Detection', 'Image Augmentation', 'Yolo'],
  install_requires = [            # I get to this in a second
          'numpy',
          'scipy',
          'Pillow',
          'argparse',
          'pyclustering',
          'tensorflow',
          'sklearn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)