

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import ndimage, stats
from scipy.ndimage import label, center_of_mass
from skimage import filters, measure
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("="*70)
print("CALCIUM IMAGING ANALYSIS PIPELINE")
print("="*70)

#%% PART 1: GENERATE SYNTHETIC CALCIUM IMAGING DATA

def generate_synthetic_calcium_data(n_frames=1000, n_neurons=50, 
                                   fov_size=(128, 128), 
                                   stim_onset_frames=[100, 300, 500, 700]):
    """
    Generate synthetic calcium imaging data
    
    Parameters:
    -----------
    n_frames : int
        Number of time frames
    n_neurons : int
        Number of neurons to simulate
    fov_size : tuple
        Field of view size (height, width)
    stim_onset_frames : list
        Frame numbers where stimuli are presented
    
    Returns:
    --------
    movie : ndarray (n_frames, height, width)
        Simulated calcium imaging movie
    true_neurons : list
        List of dictionaries containing true neuron properties
    """
    
    print("\nGenerating synthetic calcium imaging data...")
    print(f"  FOV size: {fov_size}")
    print(f"  Number of frames: {n_frames}")
    print(f"  Number of neurons: {n_neurons}")
    print(f"  Stimulus onsets: {stim_onset_frames}")
    
    height, width = fov_size
    movie = np.zeros((n_frames, height, width))
    
    # Background fluorescence with slow drift
    background_level = 100
    drift = np.sin(np.linspace(0, 4*np.pi, n_frames)) * 20
    
    true_neurons = []
    
    for i in range(n_neurons):
        # Random neuron location
        y = np.random.randint(10, height-10)
        x = np.random.randint(10, width-10)
        
        # Neuron size (soma)
        radius = np.random.uniform(3, 6)
        
        # Create spatial footprint (Gaussian)
        yy, xx = np.meshgrid(range(height), range(width), indexing='ij')
        spatial_footprint = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * radius**2))
        
        # Temporal activity
        # Decide if neuron is stimulus-responsive
        is_responsive = np.random.rand() < 0.6  # 60% are responsive
        
        temporal_trace = np.zeros(n_frames)
        
        if is_responsive:
            # Responsive: transients after stimuli
            response_probability = np.random.uniform(0.5, 1.0)
            response_amplitude = np.random.uniform(0.5, 2.0)
            
            for stim_frame in stim_onset_frames:
                if np.random.rand() < response_probability:
                    # Calcium transient (exponential decay)
                    onset = stim_frame + np.random.randint(0, 5)
                    tau_rise = 3
                    tau_decay = 10
                    
                    t = np.arange(n_frames) - onset
                    transient = np.zeros(n_frames)
                    transient[t >= 0] = response_amplitude * (
                        (1 - np.exp(-t[t >= 0] / tau_rise)) * 
                        np.exp(-t[t >= 0] / tau_decay)
                    )
                    temporal_trace += transient
        else:
            # Non-responsive: spontaneous activity
            n_spontaneous = np.random.randint(2, 8)
            for _ in range(n_spontaneous):
                onset = np.random.randint(0, n_frames - 50)
                amplitude = np.random.uniform(0.3, 0.8)
                tau_decay = 10
                
                t = np.arange(n_frames) - onset
                transient = np.zeros(n_frames)
                transient[t >= 0] = amplitude * np.exp(-t[t >= 0] / tau_decay)
                temporal_trace += transient
        
        # Add baseline and noise
        baseline = np.random.uniform(0.5, 1.5)
        temporal_trace = baseline + temporal_trace
        
        # Add to movie
        for t in range(n_frames):
            movie[t] += spatial_footprint * temporal_trace[t]
        
        true_neurons.append({
            'id': i,
            'center': (y, x),
            'radius': radius,
            'spatial_footprint': spatial_footprint,
            'temporal_trace': temporal_trace,
            'is_responsive': is_responsive,
            'baseline': baseline
        })
    
    # Add background and noise
    for t in range(n_frames):
        movie[t] += background_level + drift[t]
        movie[t] += np.random.randn(height, width) * 10  # Photon noise
    
    # Add neuropil contamination (blurred version of signal)
    neuropil = np.zeros_like(movie)
    for t in range(n_frames):
        neuropil[t] = ndimage.gaussian_filter(movie[t], sigma=5) * 0.3
    movie += neuropil
    
    print(f"\n✓ Generated {n_neurons} neurons")
    print(f"  Responsive: {sum([n['is_responsive'] for n in true_neurons])}")
    print(f"  Non-responsive: {sum([not n['is_responsive'] for n in true_neurons])}")
    
    return movie, true_neurons, stim_onset_frames

# Generate data
movie, true_neurons, stim_frames = generate_synthetic_calcium_data(
    n_frames=1000,
    n_neurons=50,
    fov_size=(128, 128),
    stim_onset_frames=[100, 300, 500, 700]
)

print(f"\nMovie shape: {movie.shape}")
print(f"Movie range: [{movie.min():.1f}, {movie.max():.1f}]")


#%% PART 2: ROI EXTRACTION

def extract_rois_correlation_pnr(movie, gSig=3, min_pnr=10, min_corr=0.8):
    """
    Extract ROIs using correlation and peak-to-noise ratio
    Simplified version of CaImAn's approach
    
    Parameters:
    -----------
    movie : ndarray (n_frames, height, width)
        Calcium imaging movie
    gSig : int
        Expected neuron radius
    min_pnr : float
        Minimum peak-to-noise ratio
    min_corr : float
        Minimum local correlation
    
    Returns:
    --------
    rois : list of dicts
        Detected ROIs with spatial footprints
    """
    
    print("\n" + "="*70)
    print("PART 2: ROI Extraction")
    print("="*70)
    
    n_frames, height, width = movie.shape
    
    # Compute local correlation image
    print("\nComputing local correlation map...")
    cn_filter = np.ones((2*gSig+1, 2*gSig+1))
    cn_filter /= cn_filter.sum()
    
    # Mean image
    mean_img = np.mean(movie, axis=0)
    
    # Local correlation
    corr_img = np.zeros((height, width))
    for y in range(gSig, height-gSig):
        for x in range(gSig, width-gSig):
            center_trace = movie[:, y, x]
            neighbor_traces = []
            
            for dy in range(-gSig, gSig+1):
                for dx in range(-gSig, gSig+1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_traces.append(movie[:, y+dy, x+dx])
            
            if len(neighbor_traces) > 0:
                neighbor_traces = np.array(neighbor_traces)
                correlations = [np.corrcoef(center_trace, nt)[0, 1] 
                               for nt in neighbor_traces]
                corr_img[y, x] = np.mean(correlations)
    
    # Compute peak-to-noise ratio
    print("Computing peak-to-noise ratio...")
    pnr_img = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            trace = movie[:, y, x]
            peak = np.percentile(trace, 95)
            noise = np.std(trace)
            if noise > 0:
                pnr_img[y, x] = peak / noise
    
    # Find candidate pixels
    print("Identifying candidate pixels...")
    candidates = (corr_img > min_corr) & (pnr_img > min_pnr)
    
    # Label connected components
    labeled, n_components = label(candidates)
    
    print(f"\nFound {n_components} candidate regions")
    
    # Extract ROIs
    rois = []
    for i in range(1, n_components + 1):
        mask = labeled == i
        
        # Size filter
        area = np.sum(mask)
        if area < 10 or area > 500:  # Reasonable neuron size
            continue
        
        # Extract spatial footprint
        footprint = np.zeros((height, width))
        
        # Get bounding box
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        # Refine footprint using local correlation
        footprint[mask] = corr_img[mask]
        
        # Normalize
        if footprint.sum() > 0:
            footprint /= footprint.sum()
        
        rois.append({
            'id': len(rois),
            'mask': mask,
            'footprint': footprint,
            'center': center_of_mass(mask),
            'area': area,
            'bbox': (y_min, y_max, x_min, x_max)
        })
    
    print(f"\n✓ Extracted {len(rois)} ROIs")
    
    return rois, corr_img, pnr_img

# Extract ROIs
rois, corr_img, pnr_img = extract_rois_correlation_pnr(
    movie, gSig=4, min_pnr=8, min_corr=0.6
)


#%% PART 3: EXTRACT FLUORESCENCE TRACES

def extract_fluorescence_traces(movie, rois):
    """
    Extract raw fluorescence traces from ROIs
    
    Parameters:
    -----------
    movie : ndarray
        Calcium imaging movie
    rois : list
        List of ROI dictionaries
        
    Returns:
    --------
    traces : ndarray (n_rois, n_frames)
        Raw fluorescence traces
    """
    
    print("\n" + "="*70)
    print("PART 3: Extract Fluorescence Traces")
    print("="*70)
    
    n_frames = movie.shape[0]
    n_rois = len(rois)
    
    traces = np.zeros((n_rois, n_frames))
    
    print(f"\nExtracting traces for {n_rois} ROIs...")
    
    for i, roi in enumerate(rois):
        footprint = roi['footprint']
        
        # Weighted sum using spatial footprint
        for t in range(n_frames):
            traces[i, t] = np.sum(movie[t] * footprint)
    
    print(f"✓ Extracted traces shape: {traces.shape}")
    
    return traces

# Extract traces
raw_traces = extract_fluorescence_traces(movie, rois)


#%% PART 4: NEUROPIL CORRECTION

def compute_neuropil_traces(movie, rois, neuropil_radius=10):
    """
    Compute neuropil contamination for each ROI
    
    Parameters:
    -----------
    movie : ndarray
        Calcium imaging movie
    rois : list
        List of ROIs
    neuropil_radius : int
        Radius for neuropil region
        
    Returns:
    --------
    neuropil_traces : ndarray
        Neuropil fluorescence traces
    """
    
    print("\n" + "="*70)
    print("PART 4: Neuropil Correction")
    print("="*70)
    
    n_frames, height, width = movie.shape
    n_rois = len(rois)
    
    neuropil_traces = np.zeros((n_rois, n_frames))
    
    print(f"\nComputing neuropil for {n_rois} ROIs...")
    
    for i, roi in enumerate(rois):
        # Create neuropil mask (annulus around ROI)
        mask = roi['mask']
        
        # Dilate to get surrounding region
        dilated = ndimage.binary_dilation(mask, iterations=neuropil_radius)
        neuropil_mask = dilated & ~mask
        
        # Extract neuropil trace
        for t in range(n_frames):
            if neuropil_mask.sum() > 0:
                neuropil_traces[i, t] = np.mean(movie[t][neuropil_mask])
    
    print(f"✓ Computed neuropil traces")
    
    return neuropil_traces

def correct_neuropil(raw_traces, neuropil_traces, alpha=0.7):
    """
    Correct for neuropil contamination
    
    F_corrected = F_raw - alpha * F_neuropil
    """
    
    print(f"\nApplying neuropil correction (alpha={alpha})...")
    corrected_traces = raw_traces - alpha * neuropil_traces
    
    return corrected_traces

# Compute and correct neuropil
neuropil_traces = compute_neuropil_traces(movie, rois)
corrected_traces = correct_neuropil(raw_traces, neuropil_traces, alpha=0.7)


#%% PART 5: COMPUTE dF/F

def compute_dff(traces, percentile=8, window=500):
    """
    Compute dF/F using rolling baseline
    
    Parameters:
    -----------
    traces : ndarray
        Fluorescence traces
    percentile : float
        Percentile for baseline estimation
    window : int
        Window size for baseline estimation
        
    Returns:
    --------
    dff : ndarray
        dF/F traces
    """
    
    print("\n" + "="*70)
    print("PART 5: Compute dF/F")
    print("="*70)
    
    n_rois, n_frames = traces.shape
    dff = np.zeros_like(traces)
    
    print(f"\nComputing dF/F for {n_rois} ROIs...")
    print(f"  Baseline: {percentile}th percentile")
    print(f"  Window: {window} frames")
    
    for i in range(n_rois):
        trace = traces[i]
        
        # Rolling baseline
        baseline = np.zeros(n_frames)
        half_window = window // 2
        
        for t in range(n_frames):
            start = max(0, t - half_window)
            end = min(n_frames, t + half_window)
            baseline[t] = np.percentile(trace[start:end], percentile)
        
        # Compute dF/F
        dff[i] = (trace - baseline) / baseline
    
    print(f"✓ Computed dF/F")
    print(f"  dF/F range: [{dff.min():.2f}, {dff.max():.2f}]")
    
    return dff

# Compute dF/F
dff_traces = compute_dff(corrected_traces)


#%% PART 6: STIMULUS RESPONSE CLASSIFICATION

def classify_stimulus_responsive(dff, stim_frames, 
                                 pre_window=10, post_window=30,
                                 n_permutations=1000, alpha=0.05):
    """
    Classify cells as stimulus-responsive using permutation test
    
    Parameters:
    -----------
    dff : ndarray (n_cells, n_frames)
        dF/F traces
    stim_frames : list
        Stimulus onset frames
    pre_window : int
        Frames before stimulus for baseline
    post_window : int
        Frames after stimulus for response
    n_permutations : int
        Number of shuffles for null distribution
    alpha : float
        Significance threshold
        
    Returns:
    --------
    results : dict
        Classification results for each cell
    """
    
    print("\n" + "="*70)
    print("PART 6: Stimulus Response Classification")
    print("="*70)
    
    n_cells, n_frames = dff.shape
    n_trials = len(stim_frames)
    
    print(f"\nClassifying {n_cells} cells...")
    print(f"  Number of trials: {n_trials}")
    print(f"  Pre-stimulus window: {pre_window} frames")
    print(f"  Post-stimulus window: {post_window} frames")
    print(f"  Permutations: {n_permutations}")
    
    results = []
    
    for cell_id in range(n_cells):
        trace = dff[cell_id]
        
        # Extract trial responses
        trial_responses = []
        baselines = []
        
        for stim_frame in stim_frames:
            if stim_frame - pre_window >= 0 and stim_frame + post_window < n_frames:
                baseline = np.mean(trace[stim_frame - pre_window:stim_frame])
                response = np.mean(trace[stim_frame:stim_frame + post_window])
                
                trial_responses.append(response - baseline)
                baselines.append(baseline)
        
        # Real mean response
        real_response = np.mean(trial_responses)
        
        # Permutation test
        null_distribution = []
        
        for _ in range(n_permutations):
            # Shuffle trial labels
            shuffled_frames = np.random.choice(
                range(pre_window, n_frames - post_window),
                size=n_trials,
                replace=False
            )
            
            shuffled_responses = []
            for shuf_frame in shuffled_frames:
                baseline = np.mean(trace[shuf_frame - pre_window:shuf_frame])
                response = np.mean(trace[shuf_frame:shuf_frame + post_window])
                shuffled_responses.append(response - baseline)
            
            null_distribution.append(np.mean(shuffled_responses))
        
        null_distribution = np.array(null_distribution)
        
        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(null_distribution) >= np.abs(real_response))
        
        # Classify
        is_responsive = p_value < alpha
        
        # Compute reliability (trial-to-trial correlation)
        if len(trial_responses) > 1:
            reliability = np.corrcoef(trial_responses, trial_responses)[0, 1]
            if np.isnan(reliability):
                reliability = 0
        else:
            reliability = 0
        
        results.append({
            'cell_id': cell_id,
            'mean_response': real_response,
            'trial_responses': trial_responses,
            'p_value': p_value,
            'is_responsive': is_responsive,
            'reliability': reliability,
            'n_trials': len(trial_responses)
        })
    
    n_responsive = sum([r['is_responsive'] for r in results])
    print(f"\n✓ Classification complete")
    print(f"  Responsive cells: {n_responsive}/{n_cells} ({100*n_responsive/n_cells:.1f}%)")
    print(f"  Non-responsive: {n_cells - n_responsive}/{n_cells}")
    
    return results

# Classify cells
classification_results = classify_stimulus_responsive(
    dff_traces, 
    stim_frames,
    n_permutations=1000
)


#%% PART 7: VISUALIZATIONS

print("\n" + "="*70)
print("PART 7: Generating Visualizations")
print("="*70)

# Visualization 1: Spatial map of ROIs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mean image
ax = axes[0]
mean_img = np.mean(movie, axis=0)
ax.imshow(mean_img, cmap='gray')
ax.set_title('Mean Image', fontsize=12, fontweight='bold')
ax.axis('off')

# ROIs colored by responsiveness
ax = axes[1]
ax.imshow(mean_img, cmap='gray', alpha=0.5)

for roi, result in zip(rois, classification_results):
    y, x = roi['center']
    color = 'red' if result['is_responsive'] else 'blue'
    
    circle = plt.Circle((x, y), radius=5, 
                       color=color, fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Add cell ID
    ax.text(x, y, str(roi['id']), 
           fontsize=6, color='white', 
           ha='center', va='center',
           bbox=dict(boxstyle='circle', facecolor=color, alpha=0.7))

ax.set_title(f'ROI Map (Red=Responsive, Blue=Non-responsive)', 
            fontsize=12, fontweight='bold')
ax.axis('off')

# Correlation and PNR maps
ax = axes[2]
im = ax.imshow(corr_img, cmap='hot')
ax.set_title('Local Correlation Map', fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('calcium_roi_maps.png', dpi=150, bbox_inches='tight')
plt.show()


# Visualization 2: Example dF/F traces
fig, axes = plt.subplots(5, 2, figsize=(14, 12))
axes = axes.flatten()

# Select 10 example cells (mix of responsive and non-responsive)
responsive_cells = [r['cell_id'] for r in classification_results if r['is_responsive']]
non_responsive_cells = [r['cell_id'] for r in classification_results if not r['is_responsive']]

example_cells = (responsive_cells[:5] if len(responsive_cells) >= 5 else responsive_cells) + \
                (non_responsive_cells[:5] if len(non_responsive_cells) >= 5 else non_responsive_cells)

time_axis = np.arange(dff_traces.shape[1]) / 30.0  # Assuming 30 Hz

for idx, cell_id in enumerate(example_cells[:10]):
    ax = axes[idx]
    
    # Plot trace
    ax.plot(time_axis, dff_traces[cell_id], linewidth=1, color='black')
    
    # Mark stimulus times
    for stim_frame in stim_frames:
        ax.axvline(x=stim_frame/30.0, color='red', linestyle='--', 
                  alpha=0.5, linewidth=1)
    
    # Get result
    result = classification_results[cell_id]
    
    # Title with classification
    status = "RESPONSIVE" if result['is_responsive'] else "Non-responsive"
    color = 'red' if result['is_responsive'] else 'blue'
    
    ax.set_title(f"Cell {cell_id} - {status} (p={result['p_value']:.3f})",
                fontsize=10, fontweight='bold', color=color)
    
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('dF/F', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Example dF/F Traces (Red lines = stimulus onset)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('calcium_example_traces.png', dpi=150, bbox_inches='tight')
plt.show()


# Visualization 3: Response amplitude vs reliability scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Extract metrics
mean_responses = [r['mean_response'] for r in classification_results]
reliabilities = [r['reliability'] for r in classification_results]
is_responsive = [r['is_responsive'] for r in classification_results]
p_values = [r['p_value'] for r in classification_results]

# Scatter plot
ax = axes[0]
responsive_mask = np.array(is_responsive)

ax.scatter(np.array(mean_responses)[~responsive_mask],
          np.array(reliabilities)[~responsive_mask],
          c='blue', alpha=0.6, s=60, label='Non-responsive', edgecolors='black')

ax.scatter(np.array(mean_responses)[responsive_mask],
          np.array(reliabilities)[responsive_mask],
          c='red', alpha=0.6, s=60, label='Responsive', edgecolors='black')

ax.set_xlabel('Mean Response Amplitude (dF/F)', fontsize=12)
ax.set_ylabel('Reliability (Trial correlation)', fontsize=12)
ax.set_title('Response Amplitude vs. Reliability', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# P-value distribution
ax = axes[1]
ax.hist(p_values, bins=20, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
          label='α = 0.05')
ax.set_xlabel('P-value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('P-value Distribution (Permutation Test)', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('calcium_response_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# Visualization 4: Trial-averaged responses
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Select top 6 responsive cells
responsive_results = [r for r in classification_results if r['is_responsive']]
responsive_results = sorted(responsive_results, key=lambda x: x['mean_response'], reverse=True)

for idx, result in enumerate(responsive_results[:6]):
    ax = axes[idx]
    
    cell_id = result['cell_id']
    
    # Extract peri-stimulus traces
    pre_frames = 20
    post_frames = 50
    
    trial_traces = []
    for stim_frame in stim_frames:
        if stim_frame - pre_frames >= 0 and stim_frame + post_frames < dff_traces.shape[1]:
            trial_trace = dff_traces[cell_id, stim_frame-pre_frames:stim_frame+post_frames]
            trial_traces.append(trial_trace)
    
    trial_traces = np.array(trial_traces)
    time = (np.arange(-pre_frames, post_frames) / 30.0) * 1000  # ms
    
    # Plot individual trials
    for trial in trial_traces:
        ax.plot(time, trial, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot mean
    mean_trace = np.mean(trial_traces, axis=0)
    sem_trace = stats.sem(trial_traces, axis=0)
    
    ax.plot(time, mean_trace, color='red', linewidth=2, label='Mean')
    ax.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
                    color='red', alpha=0.3, label='SEM')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time from stimulus (ms)', fontsize=9)
    ax.set_ylabel('dF/F', fontsize=9)
    ax.set_title(f'Cell {cell_id} (resp={result["mean_response"]:.3f})',
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Trial-Averaged Responses (Top 6 Responsive Cells)',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('calcium_trial_averaged.png', dpi=150, bbox_inches='tight')
plt.show()


# Visualization 5: Summary statistics
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# 1. Responsive vs non-responsive counts
ax1 = fig.add_subplot(gs[0, 0])
counts = [sum(is_responsive), len(is_responsive) - sum(is_responsive)]
colors = ['red', 'blue']
ax1.bar(['Responsive', 'Non-responsive'], counts, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Number of Cells', fontsize=10)
ax1.set_title('Cell Classification', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for i, count in enumerate(counts):
    ax1.text(i, count + 1, str(count), ha='center', fontsize=12, fontweight='bold')

# 2. Response amplitude distribution
ax2 = fig.add_subplot(gs[0, 1])
responsive_amplitudes = [r['mean_response'] for r in classification_results if r['is_responsive']]
non_responsive_amplitudes = [r['mean_response'] for r in classification_results if not r['is_responsive']]

ax2.hist(responsive_amplitudes, bins=15, color='red', alpha=0.6, label='Responsive')
ax2.hist(non_responsive_amplitudes, bins=15, color='blue', alpha=0.6, label='Non-responsive')
ax2.set_xlabel('Mean Response Amplitude', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('Response Amplitude Distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Reliability distribution
ax3 = fig.add_subplot(gs[0, 2])
responsive_reliability = [r['reliability'] for r in classification_results if r['is_responsive']]
non_responsive_reliability = [r['reliability'] for r in classification_results if not r['is_responsive']]

ax3.hist(responsive_reliability, bins=15, color='red', alpha=0.6, label='Responsive')
ax3.hist(non_responsive_reliability, bins=15, color='blue', alpha=0.6, label='Non-responsive')
ax3.set_xlabel('Reliability', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_title('Reliability Distribution', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. ROI size distribution
ax4 = fig.add_subplot(gs[1, 0])
roi_areas = [roi['area'] for roi in rois]
ax4.hist(roi_areas, bins=20, color='purple', alpha=0.7, edgecolor='black')
ax4.set_xlabel('ROI Area (pixels)', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('ROI Size Distribution', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Correlation: response vs p-value
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(mean_responses, -np.log10(np.array(p_values) + 1e-10),
           c=['red' if r else 'blue' for r in is_responsive],
           alpha=0.6, s=50, edgecolors='black')
ax5.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Mean Response Amplitude', fontsize=10)
ax5.set_ylabel('-log10(p-value)', fontsize=10)
ax5.set_title('Volcano Plot', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. SNR vs responsiveness
ax6 = fig.add_subplot(gs[1, 2])
snr = []
for i in range(len(rois)):
    signal = np.max(dff_traces[i])
    noise = np.std(dff_traces[i])
    snr.append(signal / noise if noise > 0 else 0)

responsive_snr = [snr[i] for i in range(len(rois)) if is_responsive[i]]
non_responsive_snr = [snr[i] for i in range(len(rois)) if not is_responsive[i]]

ax6.boxplot([responsive_snr, non_responsive_snr],
           labels=['Responsive', 'Non-responsive'],
           patch_artist=True,
           boxprops=dict(facecolor='lightgray', alpha=0.7))
ax6.set_ylabel('SNR', fontsize=10)
ax6.set_title('Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Example neuropil correction
ax7 = fig.add_subplot(gs[2, :])
example_cell = 0
time_window = slice(0, 300)

ax7.plot(raw_traces[example_cell, time_window], 
        label='Raw', color='gray', linewidth=1.5, alpha=0.7)
ax7.plot(neuropil_traces[example_cell, time_window] * 0.7, 
        label='Neuropil (×0.7)', color='orange', linewidth=1.5, alpha=0.7)
ax7.plot(corrected_traces[example_cell, time_window], 
        label='Corrected', color='blue', linewidth=1.5)

for stim_frame in stim_frames:
    if stim_frame < 300:
        ax7.axvline(x=stim_frame, color='red', linestyle='--', alpha=0.3)

ax7.set_xlabel('Frame', fontsize=10)
ax7.set_ylabel('Fluorescence (a.u.)', fontsize=10)
ax7.set_title(f'Neuropil Correction Example (Cell {example_cell})', 
             fontsize=11, fontweight='bold')
ax7.legend(fontsize=9, loc='upper right')
ax7.grid(True, alpha=0.3)

plt.suptitle('Calcium Imaging Analysis: Summary Statistics',
            fontsize=14, fontweight='bold')
plt.savefig('calcium_summary_statistics.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 8: FINAL SUMMARY

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nDataset:")
print(f"  Movie dimensions: {movie.shape}")
print(f"  Duration: {movie.shape[0]/30:.1f} seconds (@ 30 Hz)")
print(f"  Number of trials: {len(stim_frames)}")

print(f"\nROI Detection:")
print(f"  Total ROIs detected: {len(rois)}")
print(f"  Mean ROI area: {np.mean([roi['area'] for roi in rois]):.1f} pixels")
print(f"  ROI area range: [{min([roi['area'] for roi in rois])}, {max([roi['area'] for roi in rois])}]")

print(f"\nStimulus Response:")
n_resp = sum(is_responsive)
print(f"  Responsive cells: {n_resp}/{len(rois)} ({100*n_resp/len(rois):.1f}%)")
print(f"  Non-responsive cells: {len(rois)-n_resp}/{len(rois)}")

if responsive_amplitudes:
    print(f"\nResponsive Cell Properties:")
    print(f"  Mean response amplitude: {np.mean(responsive_amplitudes):.3f} ± {np.std(responsive_amplitudes):.3f}")
    print(f"  Mean reliability: {np.mean(responsive_reliability):.3f} ± {np.std(responsive_reliability):.3f}")

print(f"\nStatistical Testing:")
print(f"  Permutations: 1,000 per cell")
print(f"  Significance threshold: α = 0.05")
print(f"  Median p-value (responsive): {np.median([r['p_value'] for r in classification_results if r['is_responsive']]):.4f}")
print(f"  Median p-value (non-responsive): {np.median([r['p_value'] for r in classification_results if not r['is_responsive']]):.4f}")

print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - calcium_roi_maps.png")
print("  - calcium_example_traces.png")
print("  - calcium_response_analysis.png")
print("  - calcium_trial_averaged.png")
print("  - calcium_summary_statistics.png")
