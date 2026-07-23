import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

import { askInbox, getGmailAvailability, listSyncHistory, listThreads, startGmailAuthorization } from "./api";

export function useThreads() {
  return useQuery({ queryKey: ["threads"], queryFn: listThreads, retry: false });
}

export function useSyncHistory() {
  return useQuery({
    queryKey: ["sync-history"],
    queryFn: listSyncHistory,
    // Sync runs server-side off Pub/Sub; poll so the timeline stays live.
    refetchInterval: 5_000,
    retry: false,
  });
}

export function useGmailAvailability() {
  return useQuery({
    queryKey: ["gmail-availability"],
    queryFn: getGmailAvailability,
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useAsk() {
  return useMutation({
    mutationFn: askInbox,
    onError: () => {
      toast.error("Couldn’t connect. Try again.");
    },
  });
}

export function useGmailConnect() {
  return useMutation({
    mutationFn: startGmailAuthorization,
    onSuccess: ({ authorizationUrl }) => {
      window.location.assign(authorizationUrl);
    },
  });
}
