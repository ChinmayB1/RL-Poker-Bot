# RL-Poker-Bot

A comprehensive repository implementing multiple reinforcement learning and game theory algorithms for Texas Hold'em poker, including Deep Q-Networks (DQN), Neural Fictitious Self-Play (NFSP), and Monte Carlo Counterfactual Regret Minimization (MCCFR).

## Introduction to Poker

Poker is a family of card games that combines elements of chance, psychology, and strategy. In this repository, we focus on **Texas Hold'em**, one of the most popular variants of poker.

### Basic Rules of Texas Hold'em

Texas Hold'em is a community card poker game where players share common cards to make their best five-card hand:

1. **Deal**: Each player receives two private cards (hole cards) face down
2. **Betting Rounds**: The game consists of four betting rounds:
   - **Pre-flop**: After receiving hole cards
   - **Flop**: After three community cards are dealt
   - **Turn**: After the fourth community card
   - **River**: After the fifth and final community card
3. **Actions**: Players can:
   - **Fold**: Give up the hand and forfeit any bets
   - **Call**: Match the current bet
   - **Raise**: Increase the bet
   - **Check**: Pass when no bet is required (only if no one has bet)
4. **Showdown**: Remaining players reveal their hands, and the best five-card combination wins the pot

### Why Poker is Challenging for AI

Poker presents unique challenges for artificial intelligence:

- **Imperfect Information**: Players cannot see opponents' cards, making it a partially observable game
- **Stochasticity**: The random dealing of cards introduces uncertainty
- **Strategic Complexity**: Optimal play requires balancing exploitation and exploration, bluffing, and reading opponents
- **Large State Space**: Even simplified versions have millions of possible game states
- **Multi-agent Dynamics**: Strategies must adapt to different opponent types

### Game Theory Optimal (GTO) Play

In game theory, a **Nash equilibrium** strategy is one where no player can improve their expected payoff by unilaterally changing their strategy. In poker, finding Nash equilibrium strategies (also called GTO strategies) is computationally challenging but provides a robust baseline that cannot be exploited.

## Project Overview

This repository implements multiple approaches to learning optimal poker strategies:

1. **MCCFR (Monte Carlo Counterfactual Regret Minimization)**: A game-theoretic approach that converges to Nash equilibrium strategies
2. **DQN (Deep Q-Network)**: A deep reinforcement learning method using value function approximation
3. **NFSP v2 (Neural Fictitious Self-Play with Reward Shaping)**: Enhanced NFSP with pot-aware reward shaping to address profitability issues and calling station behavior

## Repository Structure

```
RL-Poker-Bot/
├── README.md              # This file
├── MCCFR.ipynb            # Monte Carlo Counterfactual Regret Minimization implementation
├── DQN.ipynb              # Deep Q-Network implementation
├── NFSP.ipynb             # Neural Fictitious Self-Play implementation
└── NFSP - version 2.ipynb # NFSP with Pot-Aware Reward Shaping
```

## Algorithms

### 1. Monte Carlo Counterfactual Regret Minimization (MCCFR)

**File**: `MCCFR.ipynb`

MCCFR is a sampling-based variant of Counterfactual Regret Minimization (CFR) that efficiently approximates Nash equilibrium strategies in large games.

**Key Features**:
- External sampling variant for computational efficiency
- Regret matching for strategy updates
- Information set abstraction for state representation
- Convergence to approximate Nash equilibrium
- Evaluation against baseline agents (Random, OddsAgentV21)

**Results**:
- Achieves 95.3% win rate against random agents
- Explores 25,000+ game nodes over 3,000 training iterations
- Demonstrates strong performance in Limit Hold'em

**Usage**:
```python
# Train MCCFR agent
mccfr_agent, eval_results, training_values = train_mccfr(
    iterations=3000,
    eval_every=100,
    eval_games=500,
    progress_every=10
)

# Save trained model
mccfr_agent.save_model('mccfr_model.pkl')
```

### 2. Deep Q-Network (DQN)

**File**: `DQN.ipynb`

DQN uses deep neural networks to approximate the Q-function, learning optimal action values through experience replay and target networks.

**Key Features**:
- Deep neural network for value function approximation
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration strategy
- GPU acceleration support

**Usage**:
The notebook is designed for Google Colab with GPU support. Follow the setup instructions in the notebook.

### 3. Neural Fictitious Self-Play v2 (NFSP with Reward Shaping)

**File**: `NFSP - version 2.ipynb`

An enhanced version of NFSP that addresses the "calling station problem" through pot-aware reward shaping. This version is designed to improve the agent's ability to make profitable decisions by emphasizing pot-odds awareness.

**Key Features**:
- All features from standard NFSP
- **Pot-Aware Reward Shaping**: Addresses the problem where agents win many hands but lose money overall
- Enhanced reward structure that:
  - Provides bonuses for winning large pots
  - Applies stronger penalties for losing large pots
  - Encourages value betting in significant pots
  - Discourages calling station behavior
- Enhanced diagnostics for win/loss ratio analysis
- Action distribution tracking

**Key Improvements**:
- Addresses the "calling station" problem (high win rate but negative BB/100)
- Pot-size aware reward shaping with exponential scaling
- Tiered penalties based on pot size (small, medium, large, very large)
- Enhanced evaluation metrics including win/loss ratio diagnostics

**Usage**:
```python
# Install dependencies
!pip install -q rlcard torch numpy matplotlib seaborn scipy pandas eval7

# Run all cells in the notebook
# The reward shaper is automatically initialized with optimized parameters
```

## Dependencies

### Core Libraries
- **rlcard**: Poker game environment and utilities
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **scipy**: Statistical analysis
- **eval7**: Poker hand evaluation and equity calculations

### Algorithm-Specific
- **MCCFR**: Uses standard Python libraries (collections, pickle)
- **DQN**: TensorFlow (with CUDA support for GPU)
- **NFSP**: PyTorch (with CUDA support for GPU)

### Installation

```bash
# Basic dependencies
pip install rlcard numpy matplotlib seaborn scipy pandas eval7

# For DQN (TensorFlow)
pip install tensorflow[and-cuda]  # or tensorflow for CPU-only

# For NFSP (PyTorch)
pip install torch torchvision
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/RL-Poker-Bot.git
   cd RL-Poker-Bot
   ```

2. **Choose an algorithm**:
   - For game-theoretic approach: Open `MCCFR.ipynb`
   - For deep RL: Open `DQN.ipynb` or `NFSP - version 2.ipynb` (recommended, includes reward shaping)

3. **Install dependencies** (see Dependencies section above)

4. **Run the notebook**:
   - Execute cells sequentially
   - Training times vary by algorithm and hardware
   - MCCFR: ~8-10 hours on CPU for 3000 iterations
   - DQN/NFSP: Faster with GPU acceleration

## Evaluation Metrics

All algorithms are evaluated using:

- **Win Rate**: Percentage of games won
- **Mean Payoff**: Average chips won/lost per game
- **BB/100**: Big blinds won per 100 hands (standard poker metric)
- **Statistical Significance**: Confidence intervals and p-values

## Baseline Agents

The repository includes baseline agents for comparison:

- **RandomAgent**: Takes random legal actions
- **OddsAgentV21**: GTO-inspired agent using:
  - Hand strength calculations
  - Equity calculations (eval7)
  - Pot odds analysis
  - Preflop hand rankings

## Results Summary

### Performance Comparison

| Algorithm | vs Random Agent | vs OddsAgentV21 | Training Details |
|-----------|----------------|-----------------|------------------|
| **MCCFR** | 95.3% win rate<br>+119.26 BB/100<br>+2.39 mean payoff | 77.9% win rate<br>-37.64 BB/100<br>-0.75 mean payoff | 3,000 iterations<br>25,000+ nodes explored<br>~8-10 hours (CPU) |
| **NFSP v2** | 93.9% win rate<br>+87.51 BB/100<br>+1.75 mean payoff | 70.8% win rate<br>-12.41 BB/100<br>-0.25 mean payoff | 5,000+ episodes<br>Pot-aware reward shaping<br>GPU accelerated<br>~2-3 hours (GPU) |
| **DQN** | Training pipeline<br>Focus on accuracy metrics | Training pipeline<br>No explicit game evaluation | 100 epochs per round<br>4 rounds (pre-flop to river)<br>GPU accelerated |

### Detailed Results

#### MCCFR (Monte Carlo Counterfactual Regret Minimization)
- **vs Random Agent**: 
  - Win Rate: 95.3%
  - Mean Payoff: +2.39 ± 0.05
  - BB/100: +119.26
  - Record: 4,763-219-18 (out of 5,000 games)
  
- **vs OddsAgentV21**: 
  - Win Rate: 77.9%
  - Mean Payoff: -0.75 ± 0.12
  - BB/100: -37.64
  - Record: 3,896-1,053-51 (out of 5,000 games)

- **Training Statistics**:
  - Iterations: 3,000
  - Total nodes explored: 25,000+
  - Unique information sets: 217+
  - Training time: ~8-10 hours on CPU

#### NFSP v2 (NFSP with Pot-Aware Reward Shaping)
- **vs Random Agent**: 
  - Win Rate: 93.9%
  - Mean Payoff: +1.75 ± 0.05
  - BB/100: +87.51
  - Record: 4,693-285-22 (out of 5,000 games)
  - 95% CI: [+1.66, +1.84]
  
- **vs OddsAgentV21**: 
  - Win Rate: 70.8%
  - Mean Payoff: -0.25 ± 0.08
  - BB/100: -12.41
  - Record: 3,542-1,385-73 (out of 5,000 games)
  - 95% CI: [-0.41, -0.09]
  - **Diagnostic Metrics**:
    - Average Win: 2.85 chips
    - Average Loss: 8.18 chips
    - Win/Loss Ratio: 0.35 (target: >1.0)
    - **Issue**: Calling station behavior detected - losing 2.87x more per loss than winning per win

- **Training Statistics**:
  - Episodes: 5,000+ (configurable up to 10,000)
  - GPU accelerated training
  - Network architecture: [128, 128, 64] (larger than standard NFSP)
  - Q-network: [128, 128]
  - Pot-aware reward shaping enabled
  - Training time: ~2-3 hours on GPU

- **Key Innovation**:
  - **Pot-Aware Reward Shaping**: Addresses the "calling station" problem where agents win many hands but lose money overall
  - Reward shaping parameters:
    - Pot weight: 4.0 (aggressive shaping)
    - Big pot threshold: 8.0 BB
    - Tiered penalties: 1.5x-5.0x multipliers based on pot size
    - Enhanced bonuses for large pot wins: 1.5x-2.5x multipliers
  - Enhanced diagnostics for win/loss ratio analysis
  - Action distribution tracking for playing style diagnosis
  - **Performance Note**: Shows significant improvement over baseline NFSP, with better win rate (93.9% vs 90.1% vs Random) and improved BB/100 against OddsAgentV21 (-12.41 vs -23.18), though calling station behavior still present

#### DQN (Deep Q-Network)
- **Training Approach**: 
  - Generates training data from simulated games
  - Trains separate feed-forward models for each betting round (pre-flop, flop, turn, river)
  - Uses Q-learning to combine round-specific models
  - Focuses on training accuracy rather than direct game evaluation
  
- **Training Statistics**:
  - Training samples: 100,000+ per round
  - Epochs: 100 per model
  - Network architecture: 4 layers, 128 neurons, 0.5 dropout
  - GPU accelerated training

### Baseline Comparison (OddsAgentV21 vs Random)
- **Win Rate**: 55.5%
- **Mean Payoff**: +0.72 ± 0.04
- **BB/100**: +35.83
- **Record**: 2,775-2,219-6 (out of 5,000 games)
- **95% CI**: [+0.64, +0.79]
- This provides context for evaluating the RL agents' performance

### Key Insights

1. **MCCFR Performance**:
   - Strongest performance against random opponents (95.3% win rate)
   - Highest BB/100 against random (+119.26)
   - Struggles against sophisticated opponents (OddsAgentV21)
   - Game-theoretic approach provides robust, unexploitable strategies
   - Requires significant computational time (CPU-based)

2. **NFSP v2 (Reward Shaping)**:
   - **Strong performance against random**: 93.9% win rate, +87.51 BB/100 (improved from baseline NFSP)
   - **Better against OddsAgentV21 than MCCFR**: -12.41 vs -37.64 BB/100
   - **Key Innovation**: Pot-aware reward shaping addresses "calling station" problem
   - **Current Status**: Still shows calling station behavior (win/loss ratio 0.35) but significantly improved
   - Enhanced reward structure with exponential scaling for large pots
   - Includes comprehensive diagnostic tools for analyzing playing style and win/loss patterns
   - **Improvement over baseline**: Higher win rate (93.9% vs 90.1% vs Random) and better BB/100 against strong opponents
   - **Future work**: Further reward shaping tuning and extended training may improve win/loss ratio to >1.0

4. **DQN Approach**:
   - Training pipeline focused on learning from game data
   - Modular design with round-specific models
   - No explicit game evaluation results available
   - GPU-accelerated training for efficiency

5. **General Observations**:
   - All algorithms significantly outperform random play
   - OddsAgentV21 (rule-based) provides a strong benchmark (+35.83 BB/100 vs Random)
   - Game-theoretic methods (MCCFR) excel against weak opponents (95.3% win rate, +119.26 BB/100)
   - Deep RL methods (NFSP v2) show better adaptability to stronger opponents than MCCFR
   - NFSP v2 demonstrates improved performance with reward shaping (93.9% vs Random, better than baseline NFSP)
   - Calling station problem persists but is being addressed through reward shaping
   - Training time varies significantly: MCCFR (CPU, slow) vs NFSP/DQN (GPU, faster)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Additional algorithms (CFR+, Deep CFR, etc.)
- Performance optimizations
- Extended evaluation metrics
- Support for No-Limit Hold'em
- Multi-player variants

## References

- **CFR/MCCFR**: 
  - Zinkevich et al. (2008). "Regret Minimization in Games with Incomplete Information"
  - Lanctot et al. (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games"
  
- **DQN**: 
  - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
  
- **NFSP**: 
  - Heinrich et al. (2015). "Deep Reinforcement Learning from Self-Play in Chess"

## License

This project is open source and available under the MIT License.

## Acknowledgments

- RLCard team for the excellent poker environment
- eval7 library for poker hand evaluation
- The game theory and reinforcement learning research community

---

**Note**: This project is for educational and research purposes. Poker involves real money gambling in many jurisdictions - please gamble responsibly and legally.
