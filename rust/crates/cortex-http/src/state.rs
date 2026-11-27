//! Shared server state. The SQLite connection is not stored here: each
//! `/api/pulse` opens one, reads, and drops it.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::assets::AssetStore;
use crate::auth::{AuthService, Throttle};
use crate::config::ServeConfig;
use crate::proxy::ProxyClient;

pub(crate) struct Inner {
    pub config: ServeConfig,
    pub generation: Arc<AtomicU64>,
    pub auth: AuthService,
    pub assets: AssetStore,
    pub throttle: Throttle,
    pub proxy: ProxyClient,
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub inner: Arc<Inner>,
}
