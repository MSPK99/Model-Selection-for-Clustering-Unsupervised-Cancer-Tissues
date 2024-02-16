# Unsupervised Learning for Cancer Tissue Classification

## Abstract
This project aims to apply and evaluate unsupervised learning techniques for the classification of colorectal cancer tissue patches. Utilizing a dataset of 5,000 samples across nine tissue types, we explore various clustering algorithms enhanced by dimensionality reduction methods to categorize tissue samples accurately.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [Visualizations and Figures](#visualizations-and-figures)
- [Installation and Usage](#installation-and-usage)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [License](#License)
- [Contact](#contact)

## Introduction
In the field of computational pathology, the automated classification of cancer tissue samples is a transformative advancement with the potential to enhance diagnostic precision and expedite the treatment process. Our project focuses on the development and evaluation of unsupervised machine learning algorithms to classify colorectal cancer tissue patches. The dataset consists of 5,000 high-definition images, spanning nine distinct tissue types, each integral to the accurate diagnosis of cancer stages. This study harnesses the power of state-of-the-art convolutional neural networks for feature extraction, subsequently applying clustering techniques to segregate the tissue samples into meaningful groups. Our approach emphasizes the synergy between medical expertise and artificial intelligence, with the overarching goal of providing a robust analytical tool for oncologists and researchers in the domain of colorectal cancer diagnostics.

**Clustering Workflow Visualization**

![Screenshot 2024-02-15 195101](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/8d248e0c-356b-49f1-a15f-3c4d4d658e5f)

*This visualization depicts the workflow of splitting whole slide images into patches and grouping similar patches into the same clusters. It highlights the process from the initial slide image to the extraction of individual patches, followed by clustering using UMAP to visualize and group the data effectively.*

## Dataset Description
Our dataset constitutes a rich collection of 5,000 colorectal cancer tissue patches, meticulously curated to aid in the development of automated classification systems. These patches have been extracted from larger histopathological slides, commonly used in medical diagnostics to identify and analyze cancerous cells. The dataset encompasses a diverse array of tissue types, including normal tissue and various cancer stages, which are crucial for training robust machine learning models.

**Tissue Sample Types from Colorectal Cancer Dataset**
![Screenshot 2024-02-15 194643](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/73a9cd4d-4965-4d84-8191-dfa97182a17a)

*This image displays the various tissue types present in the colorectal cancer dataset, which include Adipose (ADI), Background (BACK), Debris (DEB), Lymphocytes (LYM), Mucus (MUC), Smooth Muscle (MUS), Normal Colon Mucosa (NORM), Cancer-Associated Stroma (STR), and Colorectal Adenocarcinoma Epithelium (TUM).*
In preparation for analysis, we employed advanced image processing techniques to ensure that each patch is normalized and suitable for feature extraction. This process involved resizing images, enhancing contrast, and applying filters to minimize noise and artifacts that could interfere with the learning algorithms.

The feature extraction was performed using convolutional neural networks (CNNs), namely PathologyGAN and VGG16, which are adept at distilling complex visual information into a more compact and meaningful representation. These representations are what the clustering algorithms utilize to discern patterns and categorize the tissue samples, thus laying the groundwork for sophisticated unsupervised learning models.

## Methodology
Our methodology is grounded in the application of unsupervised machine learning algorithms to perform clustering on high-dimensional data derived from colorectal cancer tissue samples. We leveraged the power of convolutional neural networks (CNNs) such as PathologyGAN and VGG16 for feature extraction. This process distilled the complex histopathological image data into a more manageable form while preserving the essential information required for accurate clustering.

### Dimensionality Reduction Techniques
To address the computational challenge posed by the high-dimensional nature of the data, we employed two dimensionality reduction techniques: Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP). PCA was utilized to reduce the feature space while retaining the variance within the data, which is crucial for distinguishing between different tissue types. UMAP was selected for its ability to maintain the local and global structure of the data, providing a more nuanced separation of clusters in a lower-dimensional space.

### Clustering Algorithms
We experimented with several clustering algorithms to find the most suitable approach for our dataset. The main algorithms we focused on were:

- **K-Means Clustering**: A centroid-based algorithm known for its speed and efficiency, which partitions the data into K distinct clusters based on feature similarity.
- **Gaussian Mixture Models (GMM)**: A probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions, providing a soft-clustering approach.
- **Hierarchical Clustering**: An approach that builds nested clusters by progressively merging or splitting existing groups based on distance metrics.

Each algorithm has its hyperparameters finely tuned through a Grid Search methodology, where we exhaustively tested combinations to identify the most effective settings. This rigorous approach ensured that we determined the optimal number of clusters and the best configuration for each model.

### Model Evaluation
To assess the performance of our clustering models, we employed several metrics such as the Silhouette score, V-measure, Davies-Bouldin score, and Adjusted Mutual Information score. These metrics provided us with quantitative measures of the quality of clustering, enabling us to compare the effectiveness of different algorithms and their configurations.

### Justification of Choices
Our methodological choices were driven by the goal of achieving the highest clustering accuracy, as precise categorization of tissue types is paramount in medical diagnostics. The use of VGG16 for feature extraction was justified by its ability to capture detailed features at a lower level, which is crucial for distinguishing cancerous tissues. The adoption of PCA and UMAP was backed by their demonstrated capability to improve clustering outcomes in high-dimensional datasets.

In summary, our methodology is characterized by a data-driven approach that combines robust preprocessing, strategic dimensionality reduction, and meticulous model tuning and evaluation. This comprehensive strategy ensured that we systematically explored the dataset's structure and uncovered the most effective clustering methods to classify the tissue samples accurately.

## Results
Our investigation into unsupervised learning for the classification of colorectal cancer tissue samples has demonstrated the profound capability of machine learning algorithms to distinguish between various tissue types. Leveraging a dataset of 5,000 colorectal cancer tissue patches, we applied dimensionality reduction techniques such as PCA and UMAP, followed by clustering algorithms including K-Means, Gaussian Mixture Models (GMM), and Agglomerative Clustering to categorize tissue samples effectively.

### Key Findings:
- **Dimensionality Reduction:** PCA and UMAP significantly enhanced the clustering process, with PCA focusing on variance retention and UMAP preserving the data's topological structure. This facilitated a clearer segregation of tissue samples into meaningful clusters.
- **Clustering Analysis:**
  - **K-Means** demonstrated efficiency in partitioning data into distinct clusters, with silhouette scores suggesting a reasonable separation between clusters across various datasets.
  - **GMM** excelled in identifying softer boundaries between clusters, offering a nuanced understanding of the dataset's structure.
  - **Agglomerative Clustering** provided an intuitive insight into the hierarchical organization of the data, successfully grouping similar tissue types.
- **Performance Metrics:** Utilizing metrics such as silhouette scores, V-measure, and Davies-Bouldin scores, we evaluated the clustering models, revealing their strengths and limitations in tackling the complexity of cancer tissue classification.
- **Optimal Clustering:** The analysis yielded insights into the optimal number of clusters for each dataset, with specific configurations of algorithms and dimensionality reduction techniques proving most effective for distinguishing cancerous from non-cancerous tissue types.
- **Visualizations and Figures:** Our study is enriched with detailed visualizations, including scatter plots and dendrograms, illustrating the clusters formed by various algorithms. These visuals not only underscore the effectiveness of our methodologies but also provide tangible proof of our models' ability to discern distinct tissue types within the dataset.

### Implications for Computational Pathology:
The results underscore the potential of unsupervised learning in revolutionizing cancer diagnosis and treatment. By accurately classifying cancer tissue samples, our project contributes to the ongoing efforts in computational pathology to enhance diagnostic accuracy and efficiency.

In conclusion, our project represents a significant step forward in the application of unsupervised learning algorithms for cancer tissue classification. The methodologies employed and the insights gained from this study pave the way for future research in computational pathology, with the potential to improve diagnostic processes and patient outcomes significantly.

## Conclusions

Our project demonstrates the significant potential of unsupervised learning algorithms in classifying colorectal cancer tissue samples, advancing the field of computational pathology. Through the application of PCA and UMAP for dimensionality reduction and the utilization of clustering algorithms like K-Means, GMM, and Agglomerative Clustering, we've shown it is possible to effectively categorize tissue samples into meaningful groups. This approach offers a promising pathway for enhancing diagnostic processes in medical fields where labeled data may be scarce or incomplete. Our findings underscore the importance of feature extraction and dimensionality reduction in managing high-dimensional data, highlighting the effectiveness of CNNs like PathologyGAN and VGG16 in this context. While challenges remain in achieving the precision of supervised learning methods, the project lays a solid foundation for future research aimed at refining and expanding the application of unsupervised learning in medical diagnostics.

## Visualizations and Figures

Our project's repository provides a series of visualizations that elucidate the clustering process and offer a visual representation of the data's underlying structure. Below are the visualizations and their descriptions:

### First 3 principal components of PathologyGAN's PCA feature

![Screenshot 2024-02-15 190618](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/1169d450-a3cd-4d5e-8f49-1153a7165331)

This 3D scatter plot showcases the first three principal components derived from PathologyGAN's PCA feature reduction, highlighting the data's variance and potential clusters.

### Histogram of feature distribution across clusters



These histograms depict the distribution of features across clusters, providing a visual summary of how different tissue types are distributed within each cluster identified by the K-Means algorithm.

### First 3 principal components of ResNet50's UMAP feature

![Screenshot 2024-02-15 191019](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/8a0d12d9-f42b-48de-bf22-b8f167257ef6)

A 3D scatter plot of the UMAP features reduced from ResNet50, illustrating the dimensionality reduction and the discernible grouping of tissue types.

### First 3 principal components of Inception's UMAP feature

![Screenshot 2024-02-15 191110](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/b4110cb9-fc5d-4e5f-8522-4740effb081d)

This visualization presents the first three principal components from Inception's UMAP features, demonstrating the effectiveness of UMAP in maintaining the structure of high-dimensional data in a reduced form.

### First 3 principal components of VGG16's UMAP feature

![Screenshot 2024-02-15 191133](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/64d0b837-b58b-4bdc-b10b-8deb9e204cc7)

The plot provides a 3D view of the UMAP features derived from VGG16, showcasing the clustering tendency and the separation of different tissue types.

### Cluster configuration by Kmeans and Louvain

![Screenshot 2024-02-15 191203](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/bcd7ccc2-7ed6-4dd7-9929-3dcfbebd76b3)

Bar charts illustrating the cluster configurations as identified by Kmeans and Louvain clustering methods, showing the percentage of each tissue type within the clusters.

### VGG16 PCA feature dimensionally reduced features plotted in 2D

![vgg16_pca_feature_dimensionally reduced features in 2d](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/assets/157824384/bd94345e-33e1-45cb-b59c-699f46be3eed)

A 2D visualization of VGG16 PCA features, displaying the dimensionality reduction outcome and highlighting the data's potential clusters in a simplified format.

To view these visualizations in full detail and to gain a comprehensive understanding of the techniques used to generate them, please refer to the project report and the Jupyter notebooks within our repository.

Note: Replace "path_to_image/..." with the actual path to your images in the repository.


## Installation and Usage

### Prerequisites
- Ensure Python 3.8+ is installed on your machine. You can download Python from the [official website](https://www.python.org/) or use a package manager like Anaconda.
- Basic knowledge of Python programming and familiarity with Jupyter Notebooks.

### Environment Setup

#### Clone the Repository
- Install the entire repository to your system.

#### Create a Virtual Environment (optional but recommended)
- Navigate to the project directory and create a virtual environment:
    ```bash
    python -m venv venv
    ```
- Activate the environment:
    - Windows: 
    ```
    venv\Scripts\activate
    ```
    - macOS/Linux: 
    ```
    source venv/bin/activate
    ```

#### Install Dependencies
- Install all necessary Python libraries using the commands below:
    ```bash
    pip install numpy matplotlib scikit-learn h5py plotly
    ```

### Data Preparation

#### Download the Feature Files
Ensure you have the following feature files downloaded and placed in the project directory:
- `pge_dim_reduced_feature.h5`
- `resnet50_dim_reduced_feature.h5`
- `inceptionv3_dim_reduced_feature.h5`
- `vgg16_dim_reduced_feature.h5`

#### Load the Features
- Load the feature vectors from the `.h5` files using h5py, a Python package that allows you to read and write HDF5 files.

### Running the Notebooks

#### Start Jupyter Notebook
- Launch Jupyter Notebook in the project directory:
   **jupyter notebook**


#### Open the Notebooks
- In the Jupyter Notebook interface, navigate to and open each `.ipynb` file.

#### Run the Analysis
- Execute each cell in the notebook to perform the clustering analysis. Make sure to run the cells in order as they may depend on the results from previous cells.

### Usage
- The notebooks are pre-configured to read the dimensionally reduced feature files for analysis.
- You can adjust the parameters for the clustering algorithms and dimensionality reduction techniques as needed to experiment with different settings.
- Visualization cells within the notebooks will generate plots and graphs to help interpret the results of the clustering.

### Contributing
To contribute to the project, create a fork of the repository, make your changes, and submit a pull request with a clear description of your modifications.

## Contributors

The following individuals have contributed to this project with their respective roles:

- **ID: 2815755M**: Took the lead in coding, resolving bugs, and troubleshooting issues related to the codebase, ensuring the smooth execution of the project's computational tasks.
- **ID: 2805927K**: Responsible for writing the report and visualizing output graphs, ensuring that the findings were clearly communicated and supported by appropriate visual data representations.
- **ID: 2395481A**: Played a crucial role in gathering information regarding various models, providing the team with valuable insights and assisting in the evaluation of the best-performing models.
- **ID: 2740593A**: Focused on identifying the best hyperparameters for the models, contributing to the fine-tuning process which is critical for the success of machine learning projects.
- **ID: 2826798W**: Helped in compiling and gathering outputs from the models, assessed the best models for the project requirements, ensuring the selection of the most suitable algorithms.

## Acknowledgments

We extend our heartfelt thanks to all who have played a part in this project's journey. Special appreciation goes to the University of Glasgow's academic and technical staff for their invaluable expertise. Acknowledgment is also due to the creators and maintainers of the machine learning tools used in our analysis, including PathologyGAN, ResNet50, InceptionV3, and VGG16. Lastly, we're grateful to our professors for their constructive criticism and feedback, which significantly shaped our case study approach.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/MSPK99/Model-Selection-for-Clustering-Unsupervised-Cancer-Tissues/blob/main/LICENSE) file for details.

## Contact

For inquiries or proposals, contact me at praneethkumar.m@yahoo.com.
