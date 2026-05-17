#[derive(Clone, Debug)]
pub struct VertexId(pub [u8; 32]);

#[derive(Clone, Debug)]
pub struct Vertex {
    pub id: VertexId,
    pub round: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum NarwhalMessage {
    Vertex(Vertex),
    Sync(u64),
}

#[derive(Clone, Debug)]
pub enum ConsensusMessage {
    Propose(u64),
    Vote(u64, [u8; 32]),
}
