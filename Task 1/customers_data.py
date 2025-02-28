import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate sample customer data
# In a real scenario, you would load your data from a file or database
def generate_sample_data(n_samples=1000):
    """Generate synthetic customer purchase behavior data"""
    data = {
        'customer_id': range(1, n_samples + 1),
        'recency': np.random.randint(1, 100, n_samples),  # days since last purchase
        'frequency': np.random.randint(1, 50, n_samples),  # number of purchases
        'monetary': np.random.normal(500, 300, n_samples),  # average purchase value
        'avg_basket_size': np.random.normal(3, 1.5, n_samples),  # average items per purchase
        'total_items': np.random.randint(1, 200, n_samples),  # total items purchased
        'discount_usage': np.random.uniform(0, 1, n_samples),  # ratio of purchases with discount
        'returns_ratio': np.random.beta(2, 10, n_samples),  # ratio of items returned
    }
    
    # Create more realistic correlations
    data['monetary'] = data['monetary'] * (0.5 + data['frequency'] / 50)  # Higher frequency often means higher spending
    data['total_items'] = data['frequency'] * data['avg_basket_size'] * np.random.normal(1, 0.1, n_samples)
    
    # Ensure all values are positive
    data['monetary'] = np.maximum(data['monetary'], 10)
    data['avg_basket_size'] = np.maximum(data['avg_basket_size'], 1)
    data['total_items'] = np.maximum(data['total_items'], 1)
    
    return pd.DataFrame(data)

# Load or generate customer data
df = generate_sample_data()
print("Sample data generated:")
print(df.head())

# 2. Prepare data for clustering
# Select features for clustering
features = ['recency', 'frequency', 'monetary', 'avg_basket_size', 
            'total_items', 'discount_usage', 'returns_ratio']
X = df[features].copy()

# Check for missing values
print("\nMissing values in each column:")
print(X.isnull().sum())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# 3. Determine optimal number of clusters using the Elbow Method
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
# Plot the Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)
plt.tight_layout()

# 4. Perform K-means clustering with the optimal number of clusters
# For this example, let's assume k=4 is optimal based on the elbow method
optimal_k = 4  # In practice, choose based on elbow method and silhouette score
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Analyze the clusters
# Calculate cluster centers and convert back to original scale
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print("\nCluster Centers:")
print(cluster_centers_df)

# Summarize clusters
cluster_summary = df.groupby('cluster').agg({
    'customer_id': 'count',
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'avg_basket_size': 'mean',
    'total_items': 'mean',
    'discount_usage': 'mean',
    'returns_ratio': 'mean'
}).rename(columns={'customer_id': 'count'})

print("\nCluster Summary:")
print(cluster_summary)

# 6. Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis')
plt.title('Customer Segments Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add cluster centers to the plot
centers_pca = pca.transform(cluster_centers_scaled)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, marker='X', c='red', edgecolors='black')

# 7. Interpret the clusters and create customer segments descriptions
def interpret_clusters(cluster_summary):
    """Generate descriptions for each customer segment based on their characteristics"""
    descriptions = []
    
    for cluster_id in range(len(cluster_summary)):
        cluster = cluster_summary.iloc[cluster_id]
        description = f"Cluster {cluster_id} ({cluster['count']} customers): "
        
        # High/Low value determination based on relative comparison to other clusters
        if cluster['monetary'] >= cluster_summary['monetary'].mean():
            value = "High-value"
        else:
            value = "Low-value"
            
        # Frequency assessment
        if cluster['frequency'] >= cluster_summary['frequency'].mean():
            frequency = "frequent"
        else:
            frequency = "infrequent"
            
        # Recency assessment (lower is better - more recent)
        if cluster['recency'] <= cluster_summary['recency'].mean():
            recency = "recent"
        else:
            recency = "lapsed"
            
        # Basket size assessment
        if cluster['avg_basket_size'] >= cluster_summary['avg_basket_size'].mean():
            basket = "large-basket"
        else:
            basket = "small-basket"
            
        # Discount sensitivity
        if cluster['discount_usage'] >= cluster_summary['discount_usage'].mean():
            discount = "discount-sensitive"
        else:
            discount = "price-insensitive"
            
        # Returns behavior
        if cluster['returns_ratio'] >= cluster_summary['returns_ratio'].mean():
            returns = "high-returns"
        else:
            returns = "low-returns"
        
        description += f"{value}, {frequency}, {recency} shoppers with {basket} sizes, {discount}, {returns}."
        descriptions.append(description)
        
    return descriptions

segment_descriptions = interpret_clusters(cluster_summary)
print("\nCustomer Segment Descriptions:")
for desc in segment_descriptions:
    print(desc)

# 8. Create radar chart to visualize cluster profiles
def radar_chart(cluster_centers_df):
    """Create a radar chart to visualize cluster profiles"""
    # Normalize the data for radar chart
    radar_df = cluster_centers_df.copy()
    for col in radar_df.columns:
        radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
    
    # Create the radar chart
    categories = features
    N = len(categories)
    
    # Create angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the cluster profiles
    for i in range(len(radar_df)):
        values = radar_df.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Customer Segment Profiles', size=15)
    
    return fig

radar_fig = radar_chart(cluster_centers_df)

# 9. Save results
df.to_csv('customer_segments.csv', index=False)
print("\nSegmentation complete! Results saved to 'customer_segments.csv'")

# Function to recommend marketing strategies for each segment
def recommend_strategies(segment_descriptions):
    """Generate marketing strategy recommendations for each customer segment"""
    recommendations = []
    
    for i, desc in enumerate(segment_descriptions):
        recommendation = f"Marketing Strategies for Cluster {i}:\n"
        
        if "High-value" in desc and "frequent" in desc and "recent" in desc:
            recommendation += "- Loyalty programs to reward continued engagement\n"
            recommendation += "- Early access to new products\n"
            recommendation += "- Premium service options\n"
            recommendation += "- Cross-sell complementary premium products\n"
        
        elif "Low-value" in desc and "frequent" in desc:
            recommendation += "- Upselling campaigns to increase basket value\n"
            recommendation += "- Bundle offers to increase average order value\n"
            recommendation += "- Targeted product recommendations based on purchase history\n"
        
        elif "High-value" in desc and "infrequent" in desc:
            recommendation += "- Re-engagement campaigns with personalized offers\n"
            recommendation += "- Limited-time exclusive deals to drive more frequent purchases\n"
            recommendation += "- Subscription options for regularly purchased items\n"
        
        elif "lapsed" in desc:
            recommendation += "- Win-back campaigns with special incentives\n"
            recommendation += "- Surveys to understand reasons for decreased activity\n"
            recommendation += "- New product announcements to rekindle interest\n"
        
        if "discount-sensitive" in desc:
            recommendation += "- Strategic discounts and promotions\n"
            recommendation += "- Limited-time flash sales\n"
            recommendation += "- Volume-based discounts\n"
        
        if "price-insensitive" in desc:
            recommendation += "- Focus on product quality and exclusive features rather than discounts\n"
            recommendation += "- Premium product line promotions\n"
            recommendation += "- Value-added services\n"
        
        if "high-returns" in desc:
            recommendation += "- Better product descriptions and sizing guides\n"
            recommendation += "- Post-purchase satisfaction follow-ups\n"
            recommendation += "- Analyze common return reasons for product improvements\n"
        
        recommendations.append(recommendation)
    
    return recommendations

marketing_recommendations = recommend_strategies(segment_descriptions)
print("\nMarketing Strategy Recommendations:")
for rec in marketing_recommendations:
    print(rec)
    print("---")