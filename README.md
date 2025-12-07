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

### MCCFR Performance
- **vs Random Agent**: 95.3% win rate, +119.26 BB/100
- **vs OddsAgentV21**: 77.9% win rate, -37.64 BB/100
- **Training**: 3,000 iterations, 25,000+ nodes explored

### Key Insights
- MCCFR demonstrates strong performance against random opponents
- Game-theoretic approaches provide robust, unexploitable strategies
- Deep RL methods offer faster training but may be more exploitable

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
