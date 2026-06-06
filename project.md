 CEN454: Computer Vision and Machine Learning

**Problem Statement**

Detecting prohibited items in luggage is critical for ensuring passenger safety, especially in high-traffic environments such as airports, retail malls, and cargo terminals. Manual inspection of baggage is still widely practiced; however, this process is time-consuming, resource-intensive, and prone to human error due to fatigue and operational workload.

To address these challenges, there is a need for automated intelligent systems capable of accurately identifying and localizing potentially threatening objects within baggage images. Such systems can significantly reduce the workload of security personnel while improving consistency and reliability in threat detection.

In this project, you are required to design and develop a computer vision–based framework capable of performing the following tasks:

- Classify an input image as either **safe** or containing one of the following threat categories: 

- Gun 

- Knife 

- Shuriken 

- If a threat item is detected, **localize (segment) the region of the threat object within the baggage image**. 

**Dataset Description**

The dataset is divided into three major threat categories:

- Guns 

- Knives 

- Shuriken 

Dataset download link: **Will be shared ****soon**

**Project Requirements**

In this applied computer vision project, you are expected to design and implement a complete framework that:

- Classifies an input image into either **safe** or one of the defined threat categories (gun, knife, shuriken). 

- If a threat is detected, performs **localization of the threat object using appropriate computer vision techniques**. 

- Demonstrates understanding of classical computer vision and machine learning techniques learned in the course. 

**Learning Expectations**

This project requires students to:

- Conduct a thorough study of relevant literature in computer vision and object detection 

- Apply image processing and machine learning techniques appropriately 

- Integrate multiple components into a single functional system 

- Evaluate performance using appropriate metrics such as accuracy and localization quality 

The solution should be developed using concepts covered in the course along with relevant mathematical and programming tools to build a complete end-to-end system.

**Evaluation and Leaderboard Criteria**

**Evaluation Setup**

You will be evaluated using a **hidden test dataset** that will be released only during the evaluation session. This dataset will contain unseen baggage images with varying levels of difficulty, including changes in lighting, occlusion, and background complexity.

Each group must submit predictions in the required format within the allocated evaluation time.

**Submission Format**

Students must submit a CSV file containing predictions for the test images:

| **Image Name** | **Predicted Label** |
| --- | --- |
| img1.jpg | gun |
| img2.jpg | safe |
| img3.jpg | knife |

Where predicted labels include:

- safe 

- gun 

- knife 

- shuriken 

**Evaluation Metrics**

The final score will be computed using a weighted performance measure:

**1. Classification Performance (70%)**

- Accuracy 

- Macro F1-Score (to handle class imbalance) 

Final Classification Score=0.7×Accuracy+0.3×Macro F1

**2. Localization Performance (30%)**

If a threat is detected, the system must localize the object using bounding boxes or segmentation masks.

Localization quality will be measured using:

- **Intersection over Union (IoU)** 

- A prediction is considered correct if IoU ≥ 0.5 

Localization Score=Average IoU  all detected threat images** **

**Final Score Computation**

Final Score=0.7×Classification Score+0.3×Localization Score

**Leaderboard System**

A real-time **leaderboard will be generated after evaluation**, ranking all students/groups based on their Final Score.

| **Rank** | **Student / Group** | **Classification Score** | **Localization Score** | **Final Score** |
| --- | --- | --- | --- | --- |
| 1 | Group A | 0.91 | 0.84 | 0.89 |
| 2 | Group B | 0.88 | 0.80 | 0.86 |
| … | … | … | … | … |

**Evaluation Rules**

- Test dataset will be **unseen and undisclosed until evaluation day** 

- No internet or external dataset access during evaluation 

- Only inference (prediction) is allowed during evaluation session 

- Submission must be made within the allocated time window 

**Purpose of Leaderboard**

The leaderboard is intended to:

- Encourage fair competition among students 

- Evaluate model generalization on unseen data 

- Reward both classification accuracy and localization quality 

- Promote systematic model development over memorization 

Students are expected to ensure reproducibility of results and must be able to explain their methodology during viva/demonstration.

**SDG Alignment Table**

| **System Component** | **Description** | **SDG Alignment** | **Contribution to SDG** |
| --- | --- | --- | --- |
| Image Classification Module | Classifies baggage images into safe, gun, knife, or shuriken categories | SDG 16 | Enhances public safety through automated detection of prohibited and potentially dangerous items in security environments |
| Threat Object Localization Module | Identifies and localizes threat objects within baggage images using bounding boxes or segmentation techniques | SDG 16 | Improves accuracy and reliability of security screening systems, supporting safer public infrastructure |
| Feature Extraction & ML Pipeline | Applies classical computer vision techniques such as HOG, edges, and keypoint descriptors for classification | SDG 9 | Supports development of intelligent and efficient machine learning-based security systems |
| Automated Security Screening System | End-to-end system integrating classification and localization to reduce manual inspection | SDG 16, SDG 9 | Reduces human workload, minimizes operational errors, and improves efficiency in security operations |
| Dataset Utilization & Model Training | Utilization of labeled dataset for training and validating ML models | SDG 9 | Promotes innovation in AI-driven security analytics and infrastructure modernization |
| Hidden Test Evaluation System | Evaluation using unseen test images to assess model generalization | SDG 9 | Ensures robustness and real-world applicability of intelligent security systems |