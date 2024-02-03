from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, load_tools
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant. Your job is to answer questions about Premier League. For example:
        - Which team is first in the ranking?
        - Who is the player with most points?
        - Predict the results of the next Gameweek 22 based on information about Team Home and Team Away (e.g. stats, head to head) and players information. 
    
    A Gameweek is a set of fixtures (i.e. games) played by the teams in the Premier League.
    Rules are:
        - 3 points for a win
        - 1 point for a draw
        - 0 points for a loss        

    QUESTION: {question}
    MessagesPlaceholder: {agent_scratchpad}

    In order to get the information for answering you can use the following 2 GraphQL Query:

    1- GRAPHQL QUERY FOR EVENTS/GAMEWEEKS: it returns the results for each fixture for each event (i.e. Gameweek) and you should call it once. 
    You should ignore the fixtures with teamAScore and teamHScore equal to null.
	------------    
    {events_graphql_query}
    ------------
    2- GRAPHQL QUERY FOR TEAMS: it returns information about each team (e.g. name, strength attack home, etc.) and you should call it once.
	------------    
    {teams_graphql_query}
    ------------               
    3- GRAPHQL QUERY FOR PLAYERS: it returns information about each player (e.g. name, position, news, etc.) and you should call it once.
	------------    
    {players_graphql_query}
    ------------                   
    2- GRAPHQL QUERY FOR HEAD TO HEAD: it returns information about head to head teams. 
    This query uses 2 arguments: teamAShortName and teamHShortName. These are the short name for each team.
    You can find the short name for each team in the GraphQL query for Teams response (i.e. team.shortName).
    This query includes:
        - Head to head information from previous games between the 2 teams (e.g. teamAName, teamHName, teamAScore, teamHScore, etc.)
        - Stats information about the 2 teams (e.g. accurate_back_zone_pass, accurate_long_balls, aerial_won, defender_goals, etc.)
    ------------    
    {h2h_graphql_query}
    ------------
    """
)

events_graphql_query = """
{{
	events {{
		name
		fixtures {{
			finished
			id
			event
			kickoffTime
			teamAName
			teamAScore
			teamHName
			teamHScore
		}}
	}}
}}
"""
teams_graphql_query = """
{{
	teams {{
		name
		shortName
		strengthAttackHome
		strengthAttackAway
		strengthOverallHome
		strengthOverallAway
		strengthDefenceHome
		strengthDefenceAway
	}}
}}
"""
players_graphql_query = """
{{
	players {{
		webName
		team
		news
		position
		chanceOfPlayingNextRound
		eventPoints
		expectedGoals
		expectedGoalsPer90
		expectedAssists
		expectedAssistsPer90
		expectedGoalsConceded
		expectedGoalsConcededPer90
		expectedGoalInvolvements
		expectedGoalInvolvementsPer90
		form
		formRank
		formRankType
		valueForm
		valueSeason
		nowCost
		nowCostRank
		nowCostRankType
		pointsPerGame
		selectedByPercent
		selectedRank
		selectedRankType
		totalPoints
		transfersInEvent
		transfersOutEvent
		minutes
		goalsScored
		assists
		cleanSheets
		goalsConceded
		ownGoals
		penaltiesSaved
		penaltiesMissed
		penaltiesOrder
		influence
		influenceRank
		influenceRankType
		ownGoals
		starts
		startsPer90
		pointsPerGame
		ictIndex
		ictIndexRank
		ictIndexRankType
		directFreekicksOrder
		cornersAndIndirectFreekicksOrder
		threat
		threatRank
		threatRankType
		creativity
		creativityRank
		creativityRankType
		bps
		bonus
		redCards
		yellowCards
	}}
}}
"""
h2h_graphql_query = """
{{
	h2h(teamAShortName: {team_away_short_name}, teamHShortName: {team_home_short_name}) {{
		Gameweeks {{
			TeamAName
			ScoreTeamA
			TeamHName
			ScoreTeamH
			Kickoff
		}}
		StatsTeamA {{
			Name
			Value
			Description
		}}
		StatsTeamH {{
			Name
			Value
			Description
		}}
	}}
}}
"""

llm = ChatOpenAI(model="gpt-4-1106-preview")
# llm = ChatOpenAI(model="gpt-4")
# llm = ChatOpenAI(model="gpt-3.5-turbo-instruct")
tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://anfield-api-margostino.vercel.app/api/query",
    llm=llm,
)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=100,
    max_execution_time=300,
    return_intermediate_steps=True,
)
question = "Which team is first in the ranking?"
response = agent_executor.invoke(
    {
        "question": question,
        "events_graphql_query": events_graphql_query,
        "teams_graphql_query": teams_graphql_query,
        "players_graphql_query": players_graphql_query,
        "h2h_graphql_query": h2h_graphql_query,
    }
)
# print(response["intermediate_steps"])
