# 🖋️ ShadowChain Writer

A sophisticated CLI tool for crafting cyberpunk noir novels with advanced entity tracking, story structure analysis, and LaTeX document generation.

## 🎭 Features

### Core Functionality
- **📊 Entity Management**: Track characters, technologies, locations, MacGuffins, and organizations
- **📚 Story Structure**: Manage chapters, scenes, story arcs, and plot points
- **🕸️ Relationship Tracking**: Model complex relationships between entities
- **📈 Story Analytics**: Analyze story structure, pacing, and character development
- **📝 LaTeX Export**: Generate professional PDF documents
- **🤖 AI Assistance**: Get AI-powered suggestions for character development and plot

### Cyberpunk-Specific Features
- Pre-loaded with cyberpunk entity templates (hackers, crypto-tech, underground locations)
- Noir aesthetic in terminal output with neon colors
- Technical terminology integration
- Themes of privacy vs. surveillance, technology as weapon

## 🚀 Getting Started

### Prerequisites
- Rust 1.70+ (for compilation)
- RocksDB system libraries
- Optional: LaTeX installation for PDF generation

### Installation

1. **Install System Dependencies** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install build-essential librocksdb-dev pkg-config libssl-dev
# Optional for PDF generation:
sudo apt-get install texlive-full
```

2. **Build the CLI**:
```bash
cd shadowchain-writer
cargo build --release
```

3. **Initialize Your Story**:
```bash
./target/release/shadowchain-writer init "Shadows in the Chain" --author "Your Name"
```

## 📖 Usage Examples

### Entity Management
```bash
# Create a hacker character
shadowchain entity create --entity-type character "Elena Voss"

# Create cyberpunk technology
shadowchain entity create --entity-type tech "Nexus Veil"

# List all entities
shadowchain entity list

# Search for entities
shadowchain entity search "Elena"
```

### Story Structure
```bash
# Create chapters
shadowchain story chapter create "Digital Shadows"

# Create scenes within chapters
shadowchain story scene create "Digital Shadows" "Opening Chase"

# View story overview
shadowchain story overview
```

### Analysis and Visualization
```bash
# Show story statistics
shadowchain analyze stats

# Analyze entity relationships
shadowchain analyze relationships

# Show story structure analysis
shadowchain analyze structure
```

### Export
```bash
# Generate LaTeX source
shadowchain export latex --output story.tex

# Generate PDF (requires LaTeX installation)
shadowchain export pdf --output story.pdf

# Export entities as JSON
shadowchain export entities --output entities.json
```

### Interactive Dashboard
```bash
# Launch interactive dashboard
shadowchain dashboard
```

## 🎨 Pre-loaded Story Universe

The tool comes pre-populated with entities from "Shadows in the Chain":

### Characters
- **Elena Voss**: MI6 operative burned twice over, ghost in the machine
- **Marcus Hale**: CIA analyst, crypto-hunter obsessed with taking down Elena

### Technology
- **Nexus Veil**: Underground protocol for anonymous blockchain transactions
  - SNARKs and STARKs for zero-knowledge proofs
  - IPFS for content-addressed storage
  - BitTorrent DHT for peer discovery

### Locations
- **Berlin Kreuzberg**: Graffiti-scarred district, rain-slicked streets

### MacGuffins
- **Master Node**: Genesis key that can rewrite Nexus Veil's rules

## 🏗️ Architecture

### Core Modules
- **`entities.rs`**: Entity type system and relationship modeling
- **`database.rs`**: RocksDB persistence layer with indexing
- **`story.rs`**: Story structure and analysis framework
- **`cli.rs`**: Interactive command-line interface
- **`latex.rs`**: LaTeX document generation
- **`analysis.rs`**: Story analytics and visualization
- **`ai.rs`**: AI-assisted writing tools

### Key Features
- **RocksDB Backend**: Fast, persistent storage for all story data
- **UUID-based Entities**: Unique identification for all story elements
- **Relationship Graph**: Complex entity relationship modeling
- **Three-Act Analysis**: Automated story structure analysis
- **Cyberpunk Templates**: Genre-specific entity creation helpers

## 🤖 AI Features

### Character Generation
```bash
shadowchain ai character "New Character Name"
```
- Generates cyberpunk archetypes (Hacker, Corporate Agent, Street Samurai)
- Suggests character traits and motivations
- Provides cyberpunk-specific elements

### Plot Development
```bash
shadowchain ai plot
```
- Analyzes current story state
- Suggests plot development opportunities
- Identifies underutilized entities
- Recommends tension-building scenarios

### Writing Style Analysis
```bash
shadowchain ai style
```
- Analyzes sentence structure and vocabulary
- Provides reading level assessment
- Suggests style improvements
- Maintains cyberpunk noir aesthetic

## 📊 Analytics Dashboard

The tool provides comprehensive story analytics:

### Story Statistics
- Word count progress tracking
- Chapter and scene counts
- Completion percentages
- Three-act structure balance

### Entity Analytics
- Entity type distribution
- Relationship network analysis
- Most connected characters
- Orphaned entity identification

### Relationship Visualization
- Character relationship matrices
- Connection strength analysis
- Network centrality metrics
- Isolation warnings

## 🎯 Workflow Integration

### Daily Writing Process
1. **Plan**: Use `shadowchain analyze structure` to review progress
2. **Create**: Add new entities and scenes as needed
3. **Write**: Use scene management to organize content
4. **Review**: Run analytics to identify areas for development
5. **Export**: Generate PDFs for review and sharing

### Story Development Lifecycle
1. **Initialization**: Create core entities and story framework
2. **Development**: Build out characters, locations, and plot elements
3. **Analysis**: Use AI suggestions and analytics to refine story
4. **Refinement**: Adjust pacing and structure based on analysis
5. **Publication**: Export polished LaTeX/PDF documents

## 🔧 Advanced Configuration

### Database Location
```bash
shadowchain --database ./custom-story.db init "My Story"
```

### Feature Flags
The tool supports optional features:
- `ai`: AI-assisted writing tools
- `advanced-viz`: Enhanced visualization capabilities

### LaTeX Styling
The generated LaTeX uses a custom cyberpunk theme:
- Neon color schemes (blue, green, magenta)
- Technical typography
- Dark background elements
- Matrix-inspired aesthetics

## 🐛 Troubleshooting

### Common Issues
1. **RocksDB compilation errors**: Install librocksdb-dev
2. **LaTeX not found**: Install texlive or specify custom path
3. **Build timeouts**: Use `cargo build --release` with patience for first build

### Performance Tips
- Use SSD storage for RocksDB database
- Enable release mode for production use
- Clean old databases: `rm -rf story.db` to start fresh

## 📚 Story Template: "Shadows in the Chain"

The tool is specifically designed for the cyberpunk noir story "Shadows in the Chain":

**Genre**: Cyberpunk Noir Thriller
**Setting**: Near-future Berlin and global locations
**Themes**: Privacy vs. Surveillance, Technology as weapon, Identity and masks
**Time Period**: 2030s
**POV**: Third person limited
**Target**: Adult audience, ~80,000 words

The story follows Elena Voss, a burned MI6 operative, as she navigates the underground world of anonymous cryptocurrency networks while being hunted by CIA analyst Marcus Hale in a high-stakes game of digital cat and mouse.

## 🚀 Future Enhancements

### Planned Features
- Web-based dashboard interface
- Real-time collaboration support
- Advanced visualization with D3.js
- Integration with writing tools (Scrivener, etc.)
- Enhanced AI models for genre-specific suggestions
- Multi-language support for international stories

### Contributing
This tool is part of the Q-NarwhalKnight quantum consensus project but stands as an independent creative writing tool. Contributions welcome for:
- Additional genre templates
- Enhanced analytics algorithms
- UI/UX improvements
- AI model integrations

---

**Created for crafting the cyberpunk noir masterpiece "Shadows in the Chain" - where technology meets storytelling in the digital underground.** 🌃⚡🖤