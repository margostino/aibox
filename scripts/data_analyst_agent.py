import datetime
import random
from datetime import datetime

from datadog_api_client import Configuration, ApiClient
from datadog_api_client.v1.api.metrics_api import MetricsApi
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)


# llm = ChatOpenAI(model="gpt-4", temperature=0)


def convert_datetime_to_epoch(dt: str) -> str:
    dt_object = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    epoch = int(dt_object.timestamp())
    return str(epoch)


def normalize_seconds_to_milliseconds(seconds: float) -> float:
    return seconds * 1000


def normalize_percentage(percentage: float) -> float:
    return percentage * 100


def normalize_bytes_to_gigabytes(bytes: float) -> float:
    # convert bytes to gigabytes
    return bytes / 1000000000


def identity(x):
    return x


def aggregate_by_average_day(values_by_day: dict) -> dict:
    average_by_day = {}
    for key, values in values_by_day.items():
        average_by_day[key] = sum(values) / len(values)
    return average_by_day


def aggregate_by_last_value(values_by_day: dict) -> dict:
    last_value_by_day = {}
    for key, values in values_by_day.items():
        last_value_by_day[key] = values[-1]
    return last_value_by_day


def aggregate_timeseries(data_points: list, normalizer, aggregator) -> dict:
    metrics = {}
    try:
        for data_point in data_points:
            # tag = list(filter(lambda x: x.startswith("user:"), serie['tag_set']))
            tags = data_point['tag_set']
            point_list = data_point['pointlist']
            values_by_day = {}
            for point in point_list:
                epoch_point = int(point.value[0])
                if point.value[1] is not None:
                    point_value = float(point.value[1])
                    dt = datetime.fromtimestamp(epoch_point / 1000.0)
                    # key = dt.strftime('%Y-%m-%d %H')
                    key = dt.strftime('%Y-%m-%d')
                    if key in values_by_day:
                        values_by_day[key].append(float(point_value))
                    else:
                        values_by_day[key] = [float(point_value)]

            aggregated_values = aggregator(values_by_day)

            if len(tags) > 0:
                key_tag = tags[0]
            else:
                key_tag = "app"

            for day, value in aggregated_values.items():
                normalized_value = normalizer(value)

                if key_tag in metrics:
                    metrics[key_tag][day] = normalized_value
                else:
                    metrics[key_tag] = {day: normalized_value}

    except Exception as e:
        print(f"error {e}")

    return metrics


def aggregate_counter(data_points: list) -> dict:
    metrics = {}
    try:
        for data_point in data_points:
            tags = data_point['tag_set']
            if len(tags) > 0:
                key_tag = tags[0]
                last_point = data_point['pointlist'][-1]
                metrics[key_tag] = last_point.value[1]
    except Exception as e:
        print(f"error {e}")

    return metrics


def fetch_timeseries_metrics(environment: str, from_epoch: str, to_epoch: str, query) -> list:
    # print(f"environment {environment}, from_epoch {from_epoch}, to_epoch {to_epoch}")
    configuration = Configuration()
    with ApiClient(configuration) as api_client:
        api_instance = MetricsApi(api_client)
        response = api_instance.query_metrics(
            _from=int(from_epoch),  # int((datetime.now() + relativedelta(days=-1)).timestamp()),
            to=int(to_epoch),  # int(datetime.now().timestamp()),
            query=query.replace("{env}", environment),
        )
        return response.get("series")


def sample_from_dict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))


def merge_metrics(response_time_metrics: dict, request_metrics: dict, timeout_metrics: dict) -> dict:
    metrics = {}
    for key, data_points in response_time_metrics.items():
        for day, data_point in data_points.items():
            response_time = data_point
            timeout_count = timeout_metrics[key][day] if key in timeout_metrics else 0
            request_count = request_metrics[key][day] if key in request_metrics else 0
            if key in metrics:
                metrics[key][day] = (response_time, request_count, timeout_count)
            else:
                metrics[key] = {
                    day: (response_time, request_count, timeout_count)
                }
    return metrics


def fetch_metrics(environment: str, from_epoch: str, to_epoch: str) -> dict:
    response_time_metrics = fetch_response_time_metrics(environment, from_epoch, to_epoch)
    request_count_metrics = fetch_request_count_metric(environment, from_epoch, to_epoch)
    timeout_count_metrics = fetch_timeout_count_metric(environment, from_epoch, to_epoch)
    api_cpu_perc_metrics = fetch_api_cpu_usage_metrics(environment, from_epoch, to_epoch)
    api_jvm_heap_used_metrics = fetch_api_jvm_heap_usage_metrics(environment, from_epoch, to_epoch)

    metrics = merge_metrics(response_time_metrics, request_count_metrics, timeout_count_metrics)

    metrics_sample = sample_from_dict(metrics, 100)

    return {
        "upstream_metrics": metrics_sample,
        "api_cpu_perc_metrics": api_cpu_perc_metrics['app'],
        "api_jvm_heap_used_metrics": api_jvm_heap_used_metrics['app']
    }


def fetch_api_cpu_usage_metrics(environment: str, from_epoch: str, to_epoch: str) -> dict:
    query = "max:system.cpu.user{environment:{env},service:some}"
    timeseries = fetch_timeseries_metrics(environment=environment, from_epoch=from_epoch, to_epoch=to_epoch, query=query)
    return aggregate_timeseries(timeseries, identity, aggregate_by_average_day)


def fetch_api_jvm_heap_usage_metrics(environment: str, from_epoch: str, to_epoch: str) -> dict:
    query = "..."
    timeseries = fetch_timeseries_metrics(environment=environment, from_epoch=from_epoch, to_epoch=to_epoch, query=query)
    return aggregate_timeseries(timeseries, normalize_bytes_to_gigabytes, aggregate_by_average_day)


def fetch_response_time_metrics(environment: str, from_epoch: str, to_epoch: str) -> dict:
    query = "..."
    timeseries = fetch_timeseries_metrics(environment=environment, from_epoch=from_epoch, to_epoch=to_epoch, query=query)
    return aggregate_timeseries(timeseries, normalize_seconds_to_milliseconds, aggregate_by_average_day)


def fetch_request_count_metric(environment: str, from_epoch: str, to_epoch: str) -> dict:
    query = "..."
    timeseries = fetch_timeseries_metrics(environment=environment, from_epoch=from_epoch, to_epoch=to_epoch, query=query)
    # return aggregate_counter(timeseries)
    return aggregate_timeseries(timeseries, identity, aggregate_by_last_value)


def fetch_timeout_count_metric(environment: str, from_epoch: str, to_epoch: str) -> dict:
    query = "...."
    timeseries = fetch_timeseries_metrics(environment=environment, from_epoch=from_epoch, to_epoch=to_epoch, query=query)
    # return aggregate_counter(timeseries)
    return aggregate_timeseries(timeseries, identity, aggregate_by_last_value)


def fetch_logs(environment: str, from_epoch: int, to_epoch: int) -> str:
    """Tool that fetch application logs for a given environment (e.g. eu-production) from epoch to epoch and return client requests size separated by comma."""
    print(f"environment {environment}, from_epoch {from_epoch}, to_epoch {to_epoch}")
    return f"user1 request size: 100 kb, user2 request size: 7887 kb"


data_analyst_prompt_template = """
                                Given the following metrics {response_time_metrics}, {request_count_metrics}, {timeout_count_metrics}, 
                                {api_cpu_perc_metrics} and {api_jvm_heap_used_metrics} and the system {expectations} discover patterns and come up with findings in a summary report.
                                Expected findings:
                                    - How many requests have timing out above 25% total requests/day? List only the ones above threshold (timeout rate = timeout count / request count)
                                    - Who is the user with higher response time? Just one
                                Output format:
                                ```
                                    TIMEOUT (> 25%): [...list of users...] 
                                    HIGHER RESPONSE TIME: ...user name
                                ```                                                                    
                               """

data_analyst_llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(data_analyst_prompt_template)
)


# logs_fetch_tool = StructuredTool.from_function(fetch_logs)

class Metrics(BaseModel):
    expectations: str = Field(description="should be a string with system expectations")
    response_time_metrics: dict = Field(description="should be a dict with response time metrics")
    request_count_metrics: dict = Field(description="should be a dict with request count metrics")
    timeout_count_metrics: dict = Field(description="should be a dict with timeout count metrics")
    api_cpu_perc_metrics: dict = Field(description="should be a dict with api cpu usage metrics")
    api_jvm_heap_used_metrics: dict = Field(description="should be a dict with api jvm heap used metrics")


api_jvm_heap_usage_metrics_fetcher_tool = StructuredTool.from_function(
    func=fetch_api_jvm_heap_usage_metrics,
    name="ApiJvmHeapUsedMetricsFetcher",
    description="""
    Tool that fetch JVM heap usage metrics (in Gigabytes) for a given environment from a given epoch to a given epoch                         
    """
)

api_cpu_usage_metrics_fetcher_tool = StructuredTool.from_function(
    func=fetch_api_cpu_usage_metrics,
    name="ApiCpuUsageMetricsFetcher",
    description="""
    Tool that fetch application CPU usage metrics (in % percentage) for a given environment from a given epoch to a given epoch                 
    """
)

convert_datetime_to_epoch_tool = StructuredTool.from_function(
    func=convert_datetime_to_epoch,
    name="DatetimeToEpochConverter",
    description="Tool that converts a given datetime with format YYYY-mm-ddTHH:MM:SS into an epoch"
)

fetch_metrics_tool = StructuredTool.from_function(
    func=fetch_metrics,
    name="MetricsFetcher",
    description="""
    Tool that fetch application metrics for a given environment from a given epoch to a given epoch.
    Metrics:
        - metrics tuple by day: response time (average milliseconds/day), request count (count/day), timeout count (count/day)        
        - api cpu perc (average percentage/day)
        - api jvm heap used (average percentage/day)    
    Output: JSON with api cpu usage metrics, api jvm heap usage metrics and metrics tuple by day: response time (average milliseconds/day), request count (count/day), timeout count (count/day).
    """
)

data_analyst_tool = StructuredTool.from_function(
    func=data_analyst_llm_chain.run,
    name="DataAnalyst",
    args_schema=Metrics,
    description="""
    This tool discovers patterns in metrics and come up with insights and findings in a summary report for a given system expectations.
    Input Metrics:
        - CPU usage (average percentage/day)
        - JVM memory heap usage (average gigabytes/day)            
        - Metrics tuple by day: response time (average milliseconds/day), request count (count/day), timeout count (count/day)    
    """
)

tools = [convert_datetime_to_epoch_tool, fetch_metrics_tool, data_analyst_tool]


def _handle_error(error) -> str:
    return str(error).replace("Could not parse LLM output:", "")


agent = initialize_agent(
    tools=tools,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    max_execution_time=1200,
    handle_parsing_errors=_handle_error,
    early_stopping_method="generate"
)

prompt = """
            Act as a Data Analyst to dig into a API metrics, discover patterns and come up with findings for the following parameters:
                - environment: eu-production 
                - from datetime: 2023-09-08T00:00:00
                - to datetime: 2023-09-09T00:00:00
            First you need to convert the FROM/TO datetime parameters into epoch value, then you need to fetch the metrics from the API and then you need to analyze them and finally come up with findings and insights.
            Metrics to fetch:
                - CPU usage (average percentage/day)
                - JVM memory heap usage (average gigabytes/day)            
                - Metrics tuple by day: response time (average milliseconds/day), request count (count/day), timeout count (count/day)
            Additional information about the GraphQL API:
                ....
            Expectations:
                - CPU usage always below 5%
                - JVM heap usage always below 8GB
                - Response time always below 200 milliseconds
                - Request timeouts always below 1%                                
            YOU MUST SHOW ONLY OUTLIERS AND NOT ALL THE DATA.
            Output format:
            ```
                TIMEOUT (> 25%): [...list of users...] 
                HIGHER RESPONSE TIME: ...user name
            ```            
        """

print(agent.run(prompt))
