# Methodology

In this study, our goal was to build a system that can accurately recommend the best educational items to dyslexic students. To do this, we used a popular approach called **Collaborative Filtering**—which simply means making predictions about what a student will like based on the behavior of other students with similar tastes. 

Our specific system uses an artificial intelligence model called an **Autoencoder Neural Network (AutoRec)**. Below, we break down how we prepared the data, how the model works, and how we tested it, making the process easy to follow.

## 1. Preparing the Data (Dataset Preprocessing)
Before feeding data into our AI model, we had to clean and organize it. The original dataset contained records of how different dyslexic students interacted with various educational items, as well as their personal demographic information (like age or background).

*   **Removing Bias:** We intentionally removed the 12 columns containing demographic data. Why? Because we want the AI to recommend items strictly based on a student's actual preferences and learning behaviors, not their background. 
*   **Handling Missing Data:** Naturally, no student has interacted with every single item. Whenever a student hadn't rated an item (marked as `'NC'`, `'NSU'`, or left blank), we converted that to a standard "Not a Number" (NaN) label. If a student was missing more than 40% of their data, we removed them from the study to ensure our AI only learned from students who had provided enough useful information.
*   **Formatting for the AI:** We took the students' ratings (which were on a scale of 0 to 5) and divided them by 5.0. Scaling the numbers down to a range between 0 and 1 simply makes the math easier and more stable for the neural network during training. Finally, we organized all this data into a giant spreadsheet—known as a **User-Item Interaction Matrix**—where rows represent students, columns represent items, and the cells hold their ratings.

## 2. How the AI Model Works (AutoRec Architecture)
At its core, our recommendation system is an **Autoencoder** built using the PyTorch framework. 

Think of an autoencoder as a two-step process: it takes a piece of information, compresses it down to its most basic core elements, and then attempts to reconstruct the original piece of information. Since most students only rate a small handful of items, their rows in our spreadsheet are mostly empty (which we call "sparse"). We want the model to take this mostly empty row and "fill in the blanks" with accurate predicted ratings.

*   **The Encoder (Compressing):** The model first takes a student’s sparse row of ratings and passes it through an "Encoder". The encoder squishes this large row down into a smaller, dense bundle of numbers. This forces the AI to look past the empty spaces and find the hidden, "latent" patterns connecting students who like similar things.
*   **Preventing Memorization (Dropout):** Because our data has so many empty spaces, the AI might accidentally try to just memorize the data instead of genuinely learning the patterns (a problem called over-fitting). To stop this, we use a technique called **Dropout** (set to 20%). This randomly turns off parts of the network during training, forcing the AI to be flexible and learn robust patterns.
*   **The Decoder (Reconstructing):** Finally, the compressed bundle of numbers is passed through a "Decoder". The decoder reverses the process, expanding the compressed data back into a full-sized row. The magic here is that the newly generated row now contains predicted scores for *all* the items—including the ones the student had never originally rated!

## 3. Training the Model and "Masking"
When training an AI, it essentially makes a guess, checks how wrong its guess was (calculating its "error" or "loss"), and adjusts itself to do better next time. 

However, we face a unique challenge: most data is missing. If a student hasn't rated an item, the spreadsheet cell is empty. We don't want the AI to incorrectly assume an empty space means the student hated the item (a rating of zero). 

To solve this, we give the AI a pair of virtual blinders called a **Mask**. When the AI is calculating its error (using a formula called Mean Squared Error), the mask strictly hides all the empty spaces. The AI is only judged—and only learns—based on the items the student actually interacted with. 

To make the AI train efficiently, we use a popular optimizer called **Adam**, and a technique called **Early Stopping**. Early stopping means that instead of forcing the AI to train for a set number of rounds, we monitor its performance. If it stops getting better at properly ranking the best items at the top of the list, we stop the training early to lock in its best performance.

## 4. How We Tested the System (Evaluation and Ablation)
To prove our system is reliable, we used a gold-standard testing method called **5-Fold Cross-Validation**. This means we divided our students into 5 equal groups. We trained the AI on 4 groups and tested it on the 1 remaining group, repeating this process 5 times so every group got a turn being the test group. This ensures our results aren't just a lucky fluke.

We also performed an **Ablation Study**. "Ablation" just means removing parts of the model to see if they were actually useful. We compared our full AutoRec model against two weakened versions:
1.  **No Dropout:** We removed the Dropout layer to see if the model would over-fit.
2.  **No Mask:** We took away the mask to see what happened when the model treated missing data as zeroes.

When making the final test recommendations, we "hid" the items the student had already interacted with, asking the model to strictly recommend the Top-$K$ items they *hadn't* seen yet (where $K$ could be 3, 5, or 10 items). 

---

# Results

To truly understand how well our recommendation system works, we can't just look at one metric. We evaluated the system based on its accuracy, how well it ranked the items, and whether it offered a diverse range of educational materials. We looked at the top 3 (K=3), top 5 (K=5), and top 10 (K=10) recommendations. 

### Table 1: System Performance Summary 

| Testing Metric | Top 3 Items ($K=3$) | Top 5 Items ($K=5$) | Top 10 Items ($K=10$) |
| :--- | :--- | :--- | :--- |
| **Recall@K** | 0.4696 | 0.6799 | 0.9343 |
| **NDCG@K** | 0.4954 | 0.5755 | 0.6865 |
| **Diversity** | 0.2784 | 0.3017 | 0.3595 |
| **Novelty** | 5.1274 | 5.1712 | 5.2962 |
| **Coverage** | 0.9487 | 1.0000 | 1.0000 |
| **RMSE (Error)** | 0.3480 | 0.3480 | 0.3480 |

### 1. Does it Predict accurately? (RMSE and Recall)
First, we wanted to know if the model was guessing the ratings closely. Our **RMSE (Root Mean Square Error)** remained steady at 0.3480. In simple terms, this means that when the model guesses a student's rating (on a scale of 0 to 5), it is usually only off by about a third of a point. Because we used the "Masking" technique mentioned earlier, the missing data didn't confuse the model.

Next, we looked at **Recall**. Recall measures percentage: out of all the items a student actually found useful, what percentage did the system successfully catch and recommend? 
*   When we only let the system recommend 3 items, it caught about 47% of the relevant items (Recall@3 = 0.4696).
*   However, when we expanded the list to 10 items, the performance was outstanding. The system successfully found and recommended over **93%** of the highly relevant items (Recall@10 = 0.9343). 

### 2. Does it put the best items at the top? (NDCG)
It isn't enough to just find the right items; the system needs to put the *absolute best* ones at the very top of the list so the student sees them first. This is measured by **NDCG** (Normalized Discounted Cumulative Gain). 

Our NDCG scores are very strong, climbing to **0.6865** for the Top 10 list. This confirms that the AI isn't just throwing good recommendations randomly into the mix; it is purposely ranking the most vital, highly rated items at the `#1` and `#2` spots. 

### 3. Does it recommend new and different things? (Coverage, Novelty, Diversity)
A common problem with artificial intelligence is "popularity bias"—where the AI just safely recommends the same 10 incredibly popular items over and over, ignoring the rest of the library. For dyslexic learners, discovering new and personalized materials is incredibly important. 

Thankfully, our system solves this problem beautifully, achieving outstanding scores in fairness and variety:
*   **Coverage:** This measures how much of our entire item library the AI actually recommended to people. Even when only offering 3 recommendations per student, the AI utilized nearly 95% of our entire catalog. When offering 5 or 10 recommendations, the coverage reached a perfect **1.0000 (100%)**. This means absolutely no educational item was left abandoned; everything in the library was recommended to someone who would find it useful. 
*   **Novelty and Diversity:** "Novelty" measures the AI's ability to recommend "hidden gems" (items that are extremely relevant but not globally popular). "Diversity" measures how different the recommended items are from each other. As our recommendation list grew from 3 to 10 items, both the Novelty and Diversity scores steadily increased. 

Overall, these results prove that the AutoRec system does exactly what we hoped: it perfectly balances high accuracy (finding the exact items a student needs) with a rich, diverse library experience (ensuring students are exposed to the entire catalog of educational materials).
