"""mixture-of-rerankers library — hexagonal layout (domain/ports/adapters/application).

Import hygiene: only application.collect and the live experiment services may
import reranker provider clients; domain and zero-network paths stay client-free.
"""
