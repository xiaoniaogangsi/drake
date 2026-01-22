#pragma once

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/parallelism.h"
#include "drake/common/symbolic/expression.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/solver_interface.h"
#include "drake/solvers/solver_options.h"

#include "drake/geometry/optimization/graph_of_convex_sets.h"

namespace drake {
namespace geometry {
namespace optimization {

// struct GraphOfConvexSetsOptions and GcsGraphvizOptions are defined in 
// drake/geometry/optimization/graph_of_convex_sets.h

/**
GraphOfConvexSets (GCS) implements the design pattern and optimization problems
first introduced in the paper "Shortest Paths in Graphs of Convex Sets".

"Shortest Paths in Graphs of Convex Sets" by Tobia Marcucci, Jack Umenberger,
Pablo A. Parrilo, Russ Tedrake. https://arxiv.org/abs/2101.11565

@experimental

Each vertex in the graph is associated with a convex set over continuous
variables, edges in the graph contain convex costs and constraints on these
continuous variables.  We can then formulate optimization problems over this
graph, such as the shortest path problem where each visit to a vertex also
corresponds to selecting an element from the convex set subject to the costs
and constraints.  Behind the scenes, we construct efficient mixed-integer
convex transcriptions of the graph problem using MathematicalProgram.
However, we provide the option to solve an often tight convex relaxation of the
problem with GraphOfConvexSetsOptions::convex_relaxation and employ a cheap
rounding stage which solves the convex restriction along potential paths to
find a feasible solution to the original problem.

Design note: This class avoids providing any direct access to the
MathematicalProgram that it constructs nor to the decision variables /
constraints.  The users should be able to write constraints against
"placeholder" decision variables on the vertices and edges, but these get
translated in non-trivial ways to the underlying program.

@anchor nonconvex_graph_of_convex_sets
<b>Advanced Usage: Guiding Non-convex Optimization with the
%GraphOfConvexSets</b>

Solving a GCS problem using convex relaxation involves two components:
- Convex Relaxation: The relaxation of the binary variables (edge activations)
  and perspective operations on the convex cost/constraints leads to a
  convex problem that considers the graph as a whole.
- Rounding: After solving the relaxation, a randomized rounding scheme is
  applied to obtain a feasible solution for the original problem. We interpret
  the relaxed flow variables as edge probabilities to guide the maximum
  likelyhood depth first search from the source to target vertices.
  Each rounding is calling SolveConvexRestriction.

To handle non-convex constraints, one can provide convex surrogates to the
relaxation and the true non-convex constraints to the rounding problem.
These surrogates approximate the non-convex constraints, making the relaxation
solvable as a convex optimization to guide the non-convex rounding. This can be
controlled by the Transcription enum in the AddConstraint method. We
encourage users to provide a strong convex surrogate, when possible, to better
approximate the original non-convex problem.

Users can also specify a GCS implicitly, which can be important for very large
or infinite graphs, by deriving from ImplicitGraphOfConvexSets.

@ingroup geometry_optimization
*/
class MultiAgentGraphOfConvexSets {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultiAgentGraphOfConvexSets);

  /** Specify the transcription of the optimization problem to which a
  constraint or cost should be added, or from which they should be retrieved.*/
  enum class Transcription {
    kMIP,          ///< The mixed integer formulation of the GCS problem.
    kRelaxation,   ///< The relaxation of the GCS problem.
    kRestriction,  ///< The restrction of the GCS problem where the path is
                   ///< fixed.
  };

  /** Constructs an empty graph. */
  MultiAgentGraphOfConvexSets() = default;

  virtual ~MultiAgentGraphOfConvexSets();

  class Edge;  // forward declaration.

  using VertexId = Identifier<class VertexTag>; // "using" gives an alias to the class Identifier<class VertexTag>
  using EdgeId = Identifier<class EdgeTag>;

  /** Each vertex in the graph has a corresponding ConvexSet, and a std::string
  name. */
  class Vertex final {
   public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Vertex);

    ~Vertex();

    /** Returns the unique identifier associated with this Vertex. */
    VertexId id() const { return id_; }

    /** Returns the ambient dimension of the ConvexSet. */
    int ambient_dimension() const { return set_->ambient_dimension(); }

    /** Returns the name of the vertex. */
    const std::string& name() const { return name_; }

    //----- LIZHUANG added here -----
    /** Returns the number of agents of this GCS. */
    int n_agents() const { return n_agents_; }

    /** Returns the full dimension of the ConvexSet, which is n_agents*ambient_dimension */
    int full_dimension() const { return n_agents_ * set_->ambient_dimension(); }
    //-------------------------------

    /** Returns a decision variable corresponding to an element of the
    ConvexSet, which can be used for constructing symbolic::Expression costs
    and constraints. */
    const VectorX<symbolic::Variable>& x() const { return placeholder_x_; }

    /** Returns a const reference to the underlying ConvexSet. */
    const ConvexSet& set() const { return *set_; }

    /** Adds a cost to this vertex, described by a symbolic::Expression @p e
    containing *only* elements of x() as variables.  For technical reasons
    relating to being able to "turn-off" the cost on inactive vertices, all
    costs are eventually implemented with a slack variable and a constraint:
    @verbatim
    min g(x) ⇒ min ℓ, s.t. ℓ ≥ g(x).
    @endverbatim
    You must use GetSolutionCost() to retrieve the cost of the solution, rather
    than evaluating the cost directly, in order to get consistent behavior when
    solving with the different GCS transcriptions.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.
    @note Linear costs lead to negative costs if decision variables are not
    properly constrained. Users may want to check that the solution does not
    contain negative costs.
    @returns the added cost, g(x).
    @throws std::exception if e.GetVariables() is not a subset of x().
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{expression}
    */
    solvers::Binding<solvers::Cost> AddCost(
        const symbolic::Expression& e,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Adds a cost to this vertex.  @p binding must contain *only* elements of
    x() as variables. For technical reasons relating to being able to "turn-off"
    the cost on inactive vertices, all costs are eventually implemented with a
    slack variable and a constraint:
    @verbatim
    min g(x) ⇒ min ℓ, s.t. ℓ ≥ g(x).
    @endverbatim
    You must use GetSolutionCost() to retrieve the cost of the solution, rather
    than evaluating the cost directly, in order to get consistent behavior when
    solving with the different GCS transcriptions.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.
    @note Linear costs lead to negative costs if decision variables are not
    properly constrained. Users may want to check that the solution does not
    contain negative costs.
    @returns the added cost, g(x).
    @throws std::exception if binding.variables() is not a subset of x().
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{binding}
    */
    solvers::Binding<solvers::Cost> AddCost(
        const solvers::Binding<solvers::Cost>& binding,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Adds a constraint to this vertex.
    @param f must contain *only* elements of x() as variables.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.
    @throws std::exception if f.GetFreeVariables() is not a subset of x().
    @throws std::exception if ambient_dimension() == 0.
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{formula}
    */
    solvers::Binding<solvers::Constraint> AddConstraint(
        const symbolic::Formula& f,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Adds a constraint to this vertex.
    @param binding must contain *only* elements of x() as variables.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.
    @throws std::exception if binding.variables() is not a subset of x().
    @throws std::exception if ambient_dimension() == 0.
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{binding}
    */
    solvers::Binding<solvers::Constraint> AddConstraint(
        const solvers::Binding<solvers::Constraint>& binding,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Returns costs on this vertex.
    @param used_in_transcription specifies the components of the problem from
    which the constraint should be retrieved.
    @throws std::exception if no transcription is specified.
    */
    std::vector<solvers::Binding<solvers::Cost>> GetCosts(
        const std::unordered_set<Transcription>& used_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction}) const;

    /** Returns constraints on this vertex.
    @param used_in_transcription specifies the components of the problem from
    which the constraint should be retrieved.
    @throws std::exception if no transcription is specified.
    */
    std::vector<solvers::Binding<solvers::Constraint>> GetConstraints(
        const std::unordered_set<Transcription>& used_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction}) const;

    /** Returns the sum of the costs associated with this vertex in `result`, or
    std::nullopt if no solution for this vertex is available. */
    std::optional<double> GetSolutionCost(
        const solvers::MathematicalProgramResult& result) const;

    /** Returns the cost associated with the `cost` binding on this vertex in
    `result`, or std::nullopt if no solution for this vertex is available.
    @throws std::exception if cost is not associated with this vertex. */
    std::optional<double> GetSolutionCost(
        const solvers::MathematicalProgramResult& result,
        const solvers::Binding<solvers::Cost>& cost) const;

    /** Returns the solution of x() in `result`, or std::nullopt if no solution
    for this vertex is available. std::nullopt can happen if the vertex is
    deactivated (e.g. not in the shorest path) in the solution. */
    std::optional<Eigen::VectorXd> GetSolution(
        const solvers::MathematicalProgramResult& result) const;

    const std::vector<Edge*>& incoming_edges() const { return incoming_edges_; }
    const std::vector<Edge*>& outgoing_edges() const { return outgoing_edges_; }

    //----- LIZHUANG Modified Here -----
    // New query interface functions
    /** Returns the variable component with index @p idx for agent @p agent . */
    symbolic::Variable x_at(int agent, int idx) const;
    //----------------------------------

   private:
    // Constructs a new vertex.
    Vertex(VertexId id, const ConvexSet& set, std::string name);

    //------ LIZHUANG Modified Here ------
    // An overloaded construct function for Vertex to deal with multi-agent case.
    // x will be a MatrixContinuousVariable with each row representing the original VectorContinousVariable
    Vertex(VertexId id, const ConvexSet& set, std::string name, int n_agents);
    //------------------------------------

    void AddIncomingEdge(Edge* e);
    void AddOutgoingEdge(Edge* e);
    void RemoveIncomingEdge(Edge* e);
    void RemoveOutgoingEdge(Edge* e);

    const VertexId id_{};
    const std::unique_ptr<const ConvexSet> set_;
    const std::string name_{};
    const int n_agents_{};    //LIZHUANG ADDED HERE
    const VectorX<symbolic::Variable> placeholder_x_{};
    // Note: ell_[i] is associated with costs_[i].
    solvers::VectorXDecisionVariable ell_{};
    std::vector<std::pair<solvers::Binding<solvers::Cost>,
                          std::unordered_set<Transcription>>>
        costs_{};
    std::vector<std::pair<solvers::Binding<solvers::Constraint>,
                          std::unordered_set<Transcription>>>
        constraints_;

    std::vector<Edge*> incoming_edges_{};
    std::vector<Edge*> outgoing_edges_{};

    friend class MultiAgentGraphOfConvexSets;
  };

  // Note: We think of this as a directed edge in the shortest path problem, but
  // there is nothing specific to it being a directed edge here in this class.
  /** An edge in the graph connects between vertex `u` and vertex `v`.  The
  edge also holds a list of cost and constraints associated with the continuous
  variables. */
  class Edge final {
   public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Edge);

    ~Edge();

    /** Returns the unique identifier associated with this Edge. */
    EdgeId id() const { return id_; }

    /** Returns the string name associated with this edge. */
    const std::string& name() const { return name_; }

    /** Returns a const reference to the "left" Vertex that this edge connects
    to. */
    const Vertex& u() const { return *u_; }

    /** Returns a mutable reference to the "left" Vertex that this edge connects
    to. */
    Vertex& u() { return *u_; }

    /** Returns a const reference to the "right" Vertex that this edge connects
    to. */
    const Vertex& v() const { return *v_; }

    /** Returns a mutable reference to the "right" Vertex that this edge
    connects to. */
    Vertex& v() { return *v_; }


    /** Returns the binary variable associated with this edge and a specific agent. 
    It can be used to determine whether this edge was active for this agent in the 
    solution to an optimization problem, by calling GetSolution(phi()) on a returned
    MathematicalProgramResult. 
    */
    const symbolic::Variable& phi(int agent) const {
        DRAKE_DEMAND(agent >= 0 && agent < n_agents_);
        return agents_[agent].phi_;
    }

    /** Returns the number of agents for this edge. */
    const int& n_agents() const { return n_agents_; }

    /** Returns the continuous decision variables associated with vertex `u`.
    This can be used for constructing symbolic::Expression costs and
    constraints.

    See also GetSolutionPhiXu(); using `result.GetSolution(xu())` may not
    be what you want.
    */
    const VectorX<symbolic::Variable>& xu() const { return u_->x(); }

    /** Returns the continuous decision variables associated with vertex `v`.
    This can be used for constructing symbolic::Expression costs and
    constraints.

    See also GetSolutionPhiXv(); using `result.GetSolution(xv())` may not
    be what you want.
    */
    const VectorX<symbolic::Variable>& xv() const { return v_->x(); }

    /** Creates continuous slack variables for this edge and particular `agent`, 
    appending them to an internal vector of existing slack variables. These 
    slack variables can be used in any cost or constraint on this edge only, 
    and allows for modeling more complex costs and constraints.
    */
    solvers::VectorXDecisionVariable NewSlackVariables(int agent, int rows,
                                                       const std::string& name);

    /** Adds a cost to this edge for agent @p agent, described by a symbolic::Expression @p e
    containing *only* elements of xu() and xv() as variables.  For technical
    reasons relating to being able to "turn-off" the cost on inactive edges, all
    costs are eventually implemented with a slack variable and a constraint:
    @verbatim
    min g(xu, xv) ⇒ min ℓ, s.t. ℓ ≥ g(xu,xv)
    @endverbatim
    You must use GetSolutionCost() to retrieve the cost of the solution, rather
    than evaluating the cost directly, in order to get consistent behavior when
    solving with the different GCS transcriptions.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.
    @note Linear costs lead to negative costs if decision variables are not
    properly constrained. Users may want to check that the solution does not
    contain negative costs.
    @returns the added cost, g(xu, xv).
    @throws std::exception if e.GetVariables() is not a subset of xu() ∪ xv().
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{expression}
    */
    solvers::Binding<solvers::Cost> AddCostForAgent(
        int agent,
        const symbolic::Expression& e,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    // LIZHUANG ADDED HERE
    /** Adds a cost to this edge for agent @p agent.  @p binding must contain *only* elements of
    xu() and xv() as variables. For technical reasons relating to being able to
    "turn-off" the cost on inactive edges, all costs are eventually implemented
    with a slack variable and a constraint:
    @verbatim
    min g(xu, xv) ⇒ min ℓ, s.t. ℓ ≥ g(xu,xv)
    @endverbatim
    You must use GetSolutionCost() to retrieve the cost of the solution, rather
    than evaluating the cost directly, in order to get consistent behavior when
    solving with the different GCS transcriptions.
    @param agent specifies the specific agent the cost is for.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.
    @note Linear costs lead to negative costs if decision variables are not
    properly constrained. Users may want to check that the solution does not
    contain negative costs.
    @returns the added cost, g(xu, xv).
    @throws std::exception if binding.variables() is not a subset of xu() ∪
    xv().
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{binding}
    */
    solvers::Binding<solvers::Cost> AddCostForAgent(
        int agent,
        const solvers::Binding<solvers::Cost>& binding,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Adds a constraint to this edge for a specific @p agent.
    @param f must contain *only* elements of xu() and xv() as variables.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.

    @throws std::exception if f.GetFreeVariables() is not a subset of xu() ∪
    xv().
    @throws std::exception if xu() ∪ xv() is empty, i.e., when both vertices
    have an ambient dimension of zero.
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{formula}
    */
    solvers::Binding<solvers::Constraint> AddConstraintForAgent(
        int agent,
        const symbolic::Formula& f,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Adds a constraint to this edge for a specific @p agent.
    @param agent specifies the agent this constriant is for.
    @param binding must contain *only* elements of xu() and xv() as variables.
    @param use_in_transcription specifies the components of the problem to
    which the constraint should be added.

    @throws std::exception if binding.variables() is not a subset of xu() ∪
    xv().
    @throws std::exception if xu() ∪ xv() is empty, i.e., when both vertices
    have an ambient dimension of zero.
    @throws std::exception if no transcription is specified.
    @pydrake_mkdoc_identifier{binding}
    */
    solvers::Binding<solvers::Constraint> AddConstraintForAgent(
        int agent,
        const solvers::Binding<solvers::Constraint>& binding,
        const std::unordered_set<Transcription>& use_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction});

    /** Adds a constraint on the binary variable associated with this edge and this @p agent.
    @note We intentionally do not return a binding to the constraint created by
    this call, as that would allow the caller to make nonsensical modifications
    to its bounds (i.e. requiring phi == 0.5). */
    void AddPhiConstraintForAgent(int agent, bool phi_value);

    /** Removes any constraints added to @p agent with AddPhiConstraintForAgent. */
    void ClearPhiConstraintsForAgent(int agent);

    /** Removes any constraints added to any agent with AddPhiConstraintForAgent at once*/
    void ClearPhiConstraintsForAllAgents();

    /** Returns costs on this edge. (The union of costs for all agents)
    @param used_in_transcription specifies the components of the problem from
    which the constraint should be retrieved.
    @throws std::exception if no transcription is specified.
    */
    std::vector<solvers::Binding<solvers::Cost>> GetCosts(
        const std::unordered_set<Transcription>& used_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction}) const;

    /** Returns costs on this edge for a specific @p agent.
    @param agent specifies the costs of which agent should be returned.
    @param used_in_transcription specifies the components of the problem from
    which the constraint should be retrieved.
    @throws std::exception if no transcription is specified.
    */
    std::vector<solvers::Binding<solvers::Cost>> GetCostsForAgent(
        int agent,
        const std::unordered_set<Transcription>& used_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction}) const;

    /** Returns constraints on this edge. (The union of constraints for all agnets)
    @param used_in_transcription specifies the components of the problem from
    which the constraint should be retrieved.
    @throws std::exception if no transcription is specified.
    */
    std::vector<solvers::Binding<solvers::Constraint>> GetConstraints(
        const std::unordered_set<Transcription>& used_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction}) const;

    /** Returns constraints of a specific @p agent on this edge.
    @param agent specifies the constraints of which agent should be returned. 
    @param used_in_transcription specifies the components of the problem from
    which the constraint should be retrieved.
    @throws std::exception if no transcription is specified.
    */
    std::vector<solvers::Binding<solvers::Constraint>> GetConstraintsForAgent(
        int agent, 
        const std::unordered_set<Transcription>& used_in_transcription = {
            Transcription::kMIP, Transcription::kRelaxation,
            Transcription::kRestriction}) const;

    /** Returns the sum of the costs associated with this edge and this @p agent in `result`, or
    std::nullopt if no solution for this edge is available. 
    @pydrake_mkdoc_identifier{1agentallcosts} */
    std::optional<double> GetSolutionCostForAgent(
        const solvers::MathematicalProgramResult& result,
        int agent) const;

    /** Returns the cost associated with the `cost` binding on this edge and this @p agent in
    `result`, or std::nullopt if no solution for this edge is available.
    @throws std::exception if cost is not associated with this edge. 
    @pydrake_mkdoc_identifier{1agent1cost} */
    std::optional<double> GetSolutionCostForAgent(
        const solvers::MathematicalProgramResult& result,
        const solvers::Binding<solvers::Cost>& cost,
        int agent) const;

    /** Returns the sum of the costs associated with this edge and every agent in `result`, or
    std::nullopt if no solution for this edge is available. 
    @pydrake_mkdoc_identifier{allagentsallcosts} */
    std::optional<double> GetSolutionCostForAllAgents(
        const solvers::MathematicalProgramResult& result) const;

    /** Returns the cost associated with the `cost` binding on this edge and every agent in
    `result`, or std::nullopt if no solution for this edge is available.
    @throws std::exception if cost is not associated with this edge for some agents. 
    @pydrake_mkdoc_identifier{allagents1cost}*/
    std::optional<double> GetSolutionCostForAllAgents(
        const solvers::MathematicalProgramResult& result,
        const solvers::Binding<solvers::Cost>& cost) const;

    /** Returns the vector value of the slack variables associated with ϕxᵤ and for @p agent in
    `result`, or std::nullopt if no solution for this edge is available. This
    can obtain a different value than the Vertex::GetSolution(), e.g. from
    `edge->xu().GetSolution(result)`. First, a deactivated edge (defined by Phi
    ~= 0) will return the zero vector here, while Vertex::GetSolution() will
    return std::nullopt (rather than divide by zero to recover Xu). Second, in
    the case of a loose convex relaxation, the vertex version will return the
    *averaged* value of the edge slacks for all non-zero-flow edges. */
    std::optional<Eigen::VectorXd> GetSolutionPhiXuForAgent(
        const solvers::MathematicalProgramResult& result, int agent) const;

    /** Returns the vector value of the slack variables associated with ϕxᵥ and for @p agent in
    `result`, or std::nullopt if no solution for this edge is available.
    See GetSolutionPhiXu() for more details. */
    std::optional<Eigen::VectorXd> GetSolutionPhiXvForAgent(
        const solvers::MathematicalProgramResult& result, int agent) const;

   private:
    // Constructs a new edge.
    // Edge(const EdgeId& id, Vertex* u, Vertex* v, std::string name);
    Edge(const EdgeId& id, Vertex* u, Vertex* v, std::string name, int n_agents = 1);

    const EdgeId id_{};
    Vertex* const u_{};
    Vertex* const v_{};
    symbolic::Variables allowed_vars_{};
    // symbolic::Variable phi_{};
    const std::string name_{};
    
    // LIZHUANG Modified it, add a agent layer --------------------
    struct AgentData {
      symbolic::Variable phi_{};
      // We construct placeholder variables for y and z here so that they can be
      // accessed later from a MathematicalProgramResult.  We intentionally do
      // *not* provide direct access to them for the user.
    //   const VectorX<symbolic::Variable> y_{};
    //   const VectorX<symbolic::Variable> z_{};
      VectorX<symbolic::Variable> y_{};
      VectorX<symbolic::Variable> z_{};
      
      std::unordered_map<symbolic::Variable, symbolic::Variable> x_to_yz_{};
      // Note: ell_[i] is associated with costs_[i].
      solvers::VectorXDecisionVariable ell_{};
      std::vector<std::pair<solvers::Binding<solvers::Cost>,
                          std::unordered_set<Transcription>>>
          costs_{};
      solvers::VectorXDecisionVariable slacks_{};
      std::vector<std::pair<solvers::Binding<solvers::Constraint>,
                          std::unordered_set<Transcription>>>
          constraints_;
      std::optional<bool> phi_value_{};
      // Maintain a vector of allowed vars for each agent
      symbolic::Variables allowed_vars_per_agent_{};
    
      friend class MultiAgentGraphOfConvexSets;
    };

    int n_agents_{1};
    std::vector<AgentData> agents_;

    // To be compatible with the single-agent case, add some private reference
    // ERROR: These will lead to segmentation fault, since agents_[0] is still empty now.
    // symbolic::Variable& phi_ = agents_[0].phi_;
    // VectorX<symbolic::Variable>& y_ = agents_[0].y_;
    // VectorX<symbolic::Variable>& z_ = agents_[0].z_;
    // std::unordered_map<symbolic::Variable, symbolic::Variable> x_to_yz_ = agents_[0].x_to_yz_;
    // solvers::VectorXDecisionVariable& ell_ = agents_[0].ell_;
    // std::vector<std::pair<solvers::Binding<solvers::Cost>,
    //                       std::unordered_set<Transcription>>>&
    //       costs_ = agents_[0].costs_;
    // solvers::VectorXDecisionVariable& slacks_ = agents_[0].slacks_;
    // std::vector<std::pair<solvers::Binding<solvers::Constraint>,
    //                       std::unordered_set<Transcription>>>&
    //       constraints_ = agents_[0].constraints_;
    // std::optional<bool>& phi_value_ = agents_[0].phi_value_;
    // ------------------------------------------------------------

    /** Returns the AgentData for a particular @p agent on this edge. */
    const AgentData& agent_data(int agent_id) const {
        DRAKE_DEMAND(agent_id >= 0);
        DRAKE_DEMAND(agent_id < static_cast<int>(agents_.size()));
        return agents_[agent_id];
    } //  LIZHUANG ADDED.

    friend class MultiAgentGraphOfConvexSets;
  };

  /** Returns a deep copy of this graph.
  @throws std::exception if edges have slack variables. We can add this support
  once it's needed.
  */
  std::unique_ptr<MultiAgentGraphOfConvexSets> Clone() const;

  /** Adds a vertex to the graph.  A copy of @p set is cloned and stored inside
  the graph. If @p name is empty then a default name will be provided. 
  
    @pydrake_mkdoc_identifier{singleagent}
  */
  Vertex* AddVertex(const ConvexSet& set, std::string name = "");

  //----- LIZHUANG Modified Here -----
  /** Adds a vertex to the graph.  A copy of @p set is cloned and stored inside
  the graph. If @p name is empty then a default name will be provided. 
  @p n_agents gives the number of multiple sets of variables concatenated in a vector.
  
    @pydrake_mkdoc_identifier{multiagent}
  */
  Vertex* AddVertex(const ConvexSet& set, std::string name, int n_agents);
  //----------------------------------

  /** Adds a new vertex to the graph (and assigns a new unique VertexId) by
  taking the name, costs, and constraints (but not any edges) from
  `template_vertex`. `template_vertex` does not need to be registered with this
  GCS instance; this method can be used to effectively copy a Vertex from
  another GCS instance into `this`. */
  Vertex* AddVertexFromTemplate(const Vertex& template_vertex);

  // TODO(russt): Provide helper methods to add multiple vertices which share
  // the same ConvexSet.

  // ----- LIZHUANG ADDED HERE ------
  /** Adds an edge to the graph from Vertex @p u to Vertex @p v.  The
  vertex references must refer to valid vertices in this graph. If @p name is
  empty then a default name will be provided.
  @throws std::exception if `u` or `v` are not valid vertices in this graph. 
  */
  Edge* AddEdge(Vertex* u, Vertex* v, std::string name, int n_agents);
  // --------------------------------

  /** Adds an edge to the graph from Vertex `u` to Vertex `v` (and assigns a
  new unique EdgeId), by taking the name, costs, and constraints from
  `template_edge`. `template_edge` does not need to be registered with this
  GCS instance; this method can be used to effectively copy an Edge from
  another GCS instance into `this`.
  @throws std::exception if `u` or `v` are not valid vertices in this graph.
  @throws std::exception if `u` or `v` do not match the sizes of the
  `template_edge.u()` and `template_edge.v()` vertices.
  @throws std::exception if edges have slack variables. We can add this support
  once it's needed.
  */
  Edge* AddEdgeFromTemplate(Vertex* u, Vertex* v, const Edge& template_edge);

  /** Returns the first vertex (by the order added to `this`) with the given
  name, or nullptr if no such vertex exists. */
  const Vertex* GetVertexByName(const std::string& name) const;

  /** Returns the first vertex (by the order added to `this`) with the given
  name, or nullptr if no such vertex exists. */
  Vertex* GetMutableVertexByName(const std::string& name);

  /** Returns the first edge (by the order added to `this`) with the given
  name, or nullptr if no such edge exists. */
  const Edge* GetEdgeByName(const std::string& name) const;

  /** Returns the first edge (by the order added to `this`) with the given
  name, or nullptr if no such edge exists. */
  Edge* GetMutableEdgeByName(const std::string& name);

  /** Removes vertex @p vertex from the graph as well as any edges from or to
  the vertex. Runtime is O(nₑ) where nₑ is the number of edges connected to @p
  vertex
  @pre The vertex must be part of the graph.
  */
  void RemoveVertex(Vertex* vertex);

  /** Removes edge @p edge from the graph.
  @pre The edge must be part of the graph.
  */
  void RemoveEdge(Edge* edge);

  int num_vertices() const { return vertices_.size(); }
  int num_edges() const { return edges_.size(); }

  int num_agents() const {return n_agents_;}    // LIZHUANG ADDED
  void set_num_agents(int n_agents) {n_agents_ = n_agents;} // LIZHUANG ADDED

  /** Returns mutable pointers to the vertices stored in the graph. */
  std::vector<Vertex*> Vertices();

  /** Returns pointers to the vertices stored in the graph.
  @exclude_from_pydrake_mkdoc{This overload is not bound in pydrake.} */
  std::vector<const Vertex*> Vertices() const;

  /** Returns true iff `v` is registered as a vertex with `this`.

  @pydrake_mkdoc_identifier{vertex}
  */
  bool IsValid(const Vertex& v) const;

  /** Returns mutable pointers to the edges stored in the graph. */
  std::vector<Edge*> Edges();

  /** Returns pointers to the edges stored in the graph.
  @exclude_from_pydrake_mkdoc{This overload is not bound in pydrake.} */
  std::vector<const Edge*> Edges() const;

  /** Returns true iff `e` is registered as an edge with `this`.

  @pydrake_mkdoc_identifier{edge}
  */
  bool IsValid(const Edge& e) const;

  /** Removes all constraints added to any edge with AddPhiConstraintForAgent. */
  void ClearAllPhiConstraints();

  /** Returns a Graphviz string describing the graph vertices and edges for
  a specific `agent`. 
  If `result` is supplied, then the graph will be annotated with the solution
  values, according to `options`.
  @param result the optional result from a solver.
  @param options the struct containing various options for visualization.
  @param agent the index of agent that you want to visualize. By default, 
  @p agent = -1, which indicates showing the sum of flows and costs (,etc.) 
  of all agents.
  @param active_path optionally highlights a given path in the graph. The path
  is displayed as dashed edges in red, displayed in addition to the original
  graph edges.
  */
  std::string GetGraphvizStringForAgent(
      const solvers::MathematicalProgramResult* result = nullptr,
      const GcsGraphvizOptions& options = GcsGraphvizOptions(),
      int agent = -1,
      const std::vector<const Edge*>* active_path = nullptr) const;

  /** Formulates and solves the mixed-integer convex formulation of the multi-agent
  shortest path problem on the graph, as discussed in detail in

  "Shortest Paths in Graphs of Convex Sets" by Tobia Marcucci, Jack Umenberger,
  Pablo A. Parrilo, Russ Tedrake. https://arxiv.org/abs/2101.11565

  @param sources specifies the source sets for each agent.  The solver will choose any point in
  one of those set; to start at a particular continuous state consider adding a Point
  set to the graph and using that as the source.
  @param targets specifies the target sets for each agent.  The solver will choose any point in
  one of those set.
  @param n_agents records the number of agents in this problem.
  @param options include all settings for solving the shortest path problem.
  See `GraphOfConvexSetsOptions` for further details. The following default
  options will be used if they are not provided in `options`:
  - `options.convex_relaxation = false`,
  - `options.max_rounded_paths = 0`,
  - `options.preprocessing = false`.

  @throws std::exception if any of the costs or constraints in the graph are
  incompatible with the shortest path formulation or otherwise unsupported. All
  costs must be non-negative for all values of the continuous variables.
  */
  solvers::MathematicalProgramResult SolveShortestPathForMultiAgent(
    const std::vector<Vertex*>& sources, // Vertex has DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN, so we cannot use std::vector<Vertex> directly.
    const std::vector<Vertex*>& targets,
    int n_agents,
    const GraphOfConvexSetsOptions& specified_options =
        GraphOfConvexSetsOptions()) const;

  /** Extracts a path from `source` to `target` (for a specific agent denoted by
  `agent_id`) described by the `result` returned by SolveShortestPathForAgent(),
  via depth-first search following the largest values of the edge binary variables.
  @param agent_id is the specified agent ID.
  @param n_agents is the number of agents included in this multi-agent case.
  @param tolerance defines the threshold for checking the integrality
  conditions of the binary variables for each edge. `tolerance` = 0 would
  demand that the binary variables are exactly 1 for the edges on the path.
  `tolerance` = 1 would allow the binary variables to be any value in [0, 1].
  The default value is 1e-3.
  @throws std::exception if !result.is_success() or no path from `source` to
  `target` can be found in the solution.
  */
  std::vector<const Edge*> GetSolutionPathForAgent(
      const Vertex& source, const Vertex& target, 
      const int agent_id, const int n_agents,
      const solvers::MathematicalProgramResult& result,
      double tolerance = 1e-3) const;

  /** Samples a collection of unique paths from `source` to `target`, where the
   flow values (the relaxed binary variables associated with each `Edge`)
   `flows` are interpreted as the probabilities of transitioning an edge.
   The returned paths are guaranteed to be unique, and the number of returned
   paths can be 0 if no paths are found. This function implements the first part
   of the rounding scheme put forth in Section 4.2 of "Motion Planning around
   Obstacles with Convex Optimization": https://arxiv.org/abs/2205.04422

   @param source specifies the source vertex.
   @param target specifies the target vertex.
   @param flows specifies the edge flows, which are interprested as the
   probability of transition an edge. Edge flows that are not specified are
   taken to be zero.
   @param options include all settings for sampling the paths. Specifically,
   the behavior of this function is determined through `options.rounding_seed`,
   `options.max_rounded_paths`, `options.max_rounding_trials`, and
   `options.flow_tolerance`, as described in `GraphOfConvexSetsOptions`.
   @returns A vector of paths, where each path is a vector of `Edge`s.
   @throws std::exception if options.max_rounded_path < 1.
   @pydrake_mkdoc_identifier{flows}
   */
  std::vector<std::vector<const Edge*>> SamplePaths(
      const Vertex& source, const Vertex& target,
      const std::unordered_map<const Edge*, double>& flows,
      const GraphOfConvexSetsOptions& options) const;

  /** Samples a collection of unique paths from `source` to `target` of 
   a specific agent denoted as `agent_id`, where the flow values (the relaxed 
   binary variables associated with each `Edge` and for each agent)
   `flows` are interpreted as the probabilities of transitioning an edge.
   The returned paths are guaranteed to be unique, and the number of returned
   paths can be 0 if no paths are found. This function implements the first part
   of the rounding scheme put forth in Section 4.2 of "Motion Planning around
   Obstacles with Convex Optimization": https://arxiv.org/abs/2205.04422

   @param source specifies the source vertex for a specific agent.
   @param target specifies the target vertex for a specific agent.
   @param agent_id specifies the agent ID of the flow.
   @param options include all settings for sampling the paths. Specifically,
   the behavior of this function is determined through `options.rounding_seed`,
   `options.max_rounded_paths`, `options.max_rounding_trials`, and
   `options.flow_tolerance`, as described in `GraphOfConvexSetsOptions`.
   @returns A vector of paths, where each path is a vector of `Edge`s.
   @throws std::exception if options.max_rounded_path < 1.
   @pydrake_mkdoc_identifier{result}
   */
  std::vector<std::vector<const Edge*>> SamplePathsForAgent(
      const Vertex& source, const Vertex& target, const int agent_id,
      const solvers::MathematicalProgramResult& result,
      const GraphOfConvexSetsOptions& options) const;

  /** The non-convexity in a GCS problem comes from the binary variables (phi)
  associated with the edges being active or inactive in the solution. If those
  binary variables are fixed, then the problem is convex -- this is a so-called
  "convex restriction" of the original problem.

  The convex restriction can often be solved much more efficiently than solving
  the full GCS problem with additional constraints to fix the binaries; it can
  be written using less decision variables, and needs only to include the
  vertices associated with at least one of the active edges. Decision variables
  for all other convex sets will be set to NaN.

  Note that one can specify additional non-convex constraints, which may be
  not supported by all solvers. In this case, the provided solver will throw
  an exception.

  If an @p initial_guess is provided, the solution inside this result will be
  used to set the initial guess for the convex restriction. Typically, this will
  be the result obtained by solving the convex relaxation.

  @throws std::exception if the @p initial_guess does not contain solutions for
  the decision variables required in this convex restriction.

  SolveConvexRestrictionForAgent is the agent-specific version of 
  GraphOfConvexSets::SolveConvexRestriction, with the agent specified
  by the parameter @p agent_id and the number of agents specified by @p n_agents.
  */
  solvers::MathematicalProgramResult SolveConvexRestrictionForAgent(
      const std::vector<const Edge*>& active_edges,
      const int agent_id, const int n_agents,
      const GraphOfConvexSetsOptions& options = GraphOfConvexSetsOptions(),
      const solvers::MathematicalProgramResult* initial_guess = nullptr) const;

 private: /* Facilitates testing. */
  friend class PreprocessShortestPathTest;

  // Function of ConstructPreprocessingProgram for multi-agent case.
  // Construct a prog so that it contains the variables and constriants of the 
  // preprocessing program for a given edge about a given agent.
  std::unique_ptr<solvers::MathematicalProgram> ConstructPreprocessingProgramForMultiAgent(
      EdgeId edge_id, int agent_id,
      const std::map<VertexId, std::vector<int>>& incoming_edges,
      const std::map<VertexId, std::vector<int>>& outgoing_edges,
      const std::vector<VertexId>& source_ids, 
      const std::vector<VertexId>& target_ids) const;

  // Construct a prog that can be used to solve the convex restriction for a
  // given set of active edges and for a given agent specified by `agent_id` 
  // (and optionally populate with an initial guess if one is provided).
  std::unique_ptr<solvers::MathematicalProgram> ConstructRestrictionProgramForAgent(
      const std::vector<const Edge*>& active_edges,
      const int agent_id,
      const solvers::MathematicalProgramResult* initial_guess) const;

  // Add results for additional variables of a particular agent
  // denoted by `agent_id` in `result` to make it comparable with
  // other transcriptions.
  void MakeRestrictionResultLookLikeMixedIntegerForAgent(
      const int agent_id, const int n_agents,
      const solvers::MathematicalProgram& prog,
      solvers::MathematicalProgramResult* result,
      const std::vector<const Edge*>& active_edges) const;

  // Function of PreprocessShortestPath for multi-agent case
  std::set<EdgeId> PreprocessShortestPathForMultiAgent(
    const std::vector<VertexId>& source_ids, 
    const std::vector<VertexId>& target_ids,
    const GraphOfConvexSetsOptions& options) const;

  // Adds a perspective constraint to the mathematical program to upper bound
  // the cost below a slack variable, ℓ. Specifically given a cost g(x) to
  // minimize, this method implements it with a slack variable and a constraint:
  // min g(x) ⇒ min ℓ, s.t. ℓ ≥ ϕ g(ϕx)
  // `vars` is a vector of variables to be used in the cost and constraint
  // consisting of ℓ, ϕ, and ϕ times the variables in the original cost.
  void AddPerspectiveCost(solvers::MathematicalProgram* prog,
                          const solvers::Binding<solvers::Cost>& binding,
                          const solvers::VectorXDecisionVariable& vars) const;

  // Adds a perspective version of the constraint to the mathematical program.
  // Specifically given a constraint h(x) ≤ b, this method implements its
  // perspective:
  // h(x) ≤ b ⇒ h(ϕx) ≤ ϕb
  // `vars` is a vector of variables to be used in the constraint consisting of
  // ϕ, and ϕ times the variables in the original constraint.
  void AddPerspectiveConstraint(
      solvers::MathematicalProgram* prog,
      const solvers::Binding<solvers::Constraint>& binding,
      const solvers::VectorXDecisionVariable& vars) const;

  // Note: we use VertexId and EdgeId (vs e.g. Vertex* and Edge*) here to
  // provide consistent ordering of the vertices/edges. This is important for
  // producing consistent MathematicalPrograms (the order of costs and
  // constraints can change the behavior). But prefer using Vertex* and Edge*
  // over VertexId and EdgeId in the public API; this means avoiding any sorted
  // containers (like std::set or std::map) using their default ordering.
  std::map<VertexId, std::unique_ptr<Vertex>> vertices_{};
  std::map<EdgeId, std::unique_ptr<Edge>> edges_{};

  // The member parameter for the number of agents in this MultiAgentGraphOfConvexSets
  int n_agents_{1};
};

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
