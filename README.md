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

This repository implements three different approaches to learning optimal poker strategies:

1. **MCCFR (Monte Carlo Counterfactual Regret Minimization)**: A game-theoretic approach that converges to Nash equilibrium strategies
2. **DQN (Deep Q-Network)**: A deep reinforcement learning method using value function approximation
3. **NFSP (Neural Fictitious Self-Play)**: Combines deep learning with fictitious play to learn approximate Nash equilibria

## Repository Structure

```
RL-Poker-Bot/
├── README.md           # This file
├── MCCFR.ipynb         # Monte Carlo Counterfactual Regret Minimization implementation
├── DQN.ipynb           # Deep Q-Network implementation
└── NFSP.ipynb          # Neural Fictitious Self-Play implementation
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

### 3. Neural Fictitious Self-Play (NFSP)

**File**: `NFSP.ipynb`

NFSP combines deep reinforcement learning with fictitious play, where agents learn by playing against a mixture of their past strategies.

**Key Features**:
- Dual network architecture (RL network + average strategy network)
- Reservoir sampling for opponent strategy storage
- Supervised learning for average strategy
- Reinforcement learning for best response
- Convergence to approximate Nash equilibrium

**Usage**:
```python
# Install dependencies
!pip install -q rlcard torch numpy matplotlib seaborn scipy pandas eval7

# Run all cells in the notebook
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
   - For deep RL: Open `DQN.ipynb` or `NFSP.ipynb`

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
| **NFSP** | 90.1% win rate<br>+61.10 BB/100<br>+1.22 mean payoff | 61.0% win rate<br>-23.18 BB/100<br>-0.46 mean payoff | 5,000 episodes<br>GPU accelerated<br>~2-3 hours (GPU) |
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

#### NFSP (Neural Fictitious Self-Play)
- **vs Random Agent**: 
  - Win Rate: 90.1%
  - Mean Payoff: +1.22 ± 0.03
  - BB/100: +61.10
  - Record: 4,506-457-37 (out of 5,000 games)
  
- **vs OddsAgentV21**: 
  - Win Rate: 61.0%
  - Mean Payoff: -0.46 ± 0.05
  - BB/100: -23.18
  - Record: 3,048-1,810-142 (out of 5,000 games)

- **Training Statistics**:
  - Episodes: 5,000
  - GPU accelerated training
  - Training time: ~2-3 hours on GPU

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
- **Win Rate**: 55.3% - 56.3%
- **Mean Payoff**: +0.80 ± 0.04
- **BB/100**: +39.87 - +42.33
- This provides context for evaluating the RL agents' performance

### Key Insights

1. **MCCFR Performance**:
   - Strongest performance against random opponents (95.3% win rate)
   - Highest BB/100 against random (+119.26)
   - Struggles against sophisticated opponents (OddsAgentV21)
   - Game-theoretic approach provides robust, unexploitable strategies
   - Requires significant computational time (CPU-based)

2. **NFSP Performance**:
   - Strong performance against random (90.1% win rate)
   - Better performance against OddsAgentV21 than MCCFR (-23.18 vs -37.64 BB/100)
   - Faster training with GPU acceleration
   - Balances exploitation and exploration through self-play

3. **DQN Approach**:
   - Training pipeline focused on learning from game data
   - Modular design with round-specific models
   - No explicit game evaluation results available
   - GPU-accelerated training for efficiency

4. **General Observations**:
   - All algorithms significantly outperform random play
   - OddsAgentV21 (rule-based) provides a strong benchmark
   - Game-theoretic methods (MCCFR) excel against weak opponents
   - Deep RL methods (NFSP) show better adaptability to stronger opponents
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
