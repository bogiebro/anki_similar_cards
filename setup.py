from setuptools import setup

setup(name='anki_similar_notes',
      version='0.1',
      description='Shows similar cards as you type',
      url='http://github.com/bogiebro/anki_similar_cards',
      author='Sam Anklesaria',
      license='MIT',
      install_requires=[
          'numpy', 'lxml', 'scipy', 'sklearn'
      ])
