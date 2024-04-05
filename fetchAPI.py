import requests as rq
response = rq.get('https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams?limit=32')
print(response.status_code)
listTeams = []
for i in range(32):
    response_TeamName = rq.get(response.json()['items'][i]['$ref'])
    listTeams.append(response_TeamName.json()['displayName'])

print(listTeams)



res = rq.get('https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes?limit=1000&active=true')
print(res.status_code)
print(res.json()['items'].__sizeof__())








